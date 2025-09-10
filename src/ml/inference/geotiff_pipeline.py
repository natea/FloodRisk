"""
Inference pipeline for GeoTIFF output generation.
Implementation based on APPROACH.md specifications for deliverable formats.
"""

import torch
import numpy as np
import rasterio
from rasterio import features
from rasterio.crs import CRS
from rasterio.transform import from_bounds
import xarray as xr
import geopandas as gpd
from shapely.geometry import shape
from pathlib import Path
from typing import Tuple, Optional, Dict, List, Union
from scipy import ndimage
import logging

logger = logging.getLogger(__name__)


class GeoTIFFInferencePipeline:
    """
    Generate flood prediction outputs in GeoTIFF format.

    Based on APPROACH.md deliverables:
    - flood_prob.tif (float32, 0-1)
    - flood_extent.tif (uint8; thresholded & morphology-cleaned)
    - flood_extent.gpkg (polygonized, dissolved)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        tile_size: int = 512,
        tile_overlap: int = 64,
        batch_size: int = 4,
    ):
        """
        Initialize inference pipeline.

        Args:
            model: Trained flood prediction model
            device: Device for inference
            tile_size: Size of inference tiles
            tile_overlap: Overlap between tiles
            batch_size: Batch size for inference
        """
        self.model = model
        self.device = device
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.batch_size = batch_size
        self.stride = tile_size - tile_overlap

        # Move model to device and set to eval mode
        self.model.to(device)
        self.model.eval()

        logger.info(f"Initialized GeoTIFF inference pipeline on {device}")

    def predict_large_raster(
        self,
        input_data: xr.Dataset,
        output_dir: Path,
        threshold: float = 0.5,
        apply_morphology: bool = True,
        mc_samples: Optional[int] = None,
    ) -> Dict[str, Path]:
        """
        Predict flood extent for large raster using sliding window approach.

        Args:
            input_data: Input dataset with DEM, rainfall, and derived features
            output_dir: Directory to save outputs
            threshold: Probability threshold for binary classification
            apply_morphology: Whether to apply morphological cleaning
            mc_samples: Number of MC dropout samples for uncertainty (optional)

        Returns:
            Dictionary mapping output type to file path
        """
        logger.info("Starting large raster prediction")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get raster dimensions and create output arrays
        height, width = input_data.dims["y"], input_data.dims["x"]

        # Initialize output arrays
        prob_output = np.zeros((height, width), dtype=np.float32)
        weight_output = np.zeros((height, width), dtype=np.float32)

        if mc_samples:
            uncertainty_output = np.zeros((height, width), dtype=np.float32)

        # Generate tiles for inference
        tiles = self._generate_inference_tiles(input_data)

        logger.info(f"Processing {len(tiles)} tiles for inference")

        # Process tiles in batches
        for i in range(0, len(tiles), self.batch_size):
            batch_tiles = tiles[i : i + self.batch_size]

            # Prepare batch
            batch_inputs = []
            batch_positions = []

            for tile_info in batch_tiles:
                tile_data = tile_info["data"]
                batch_inputs.append(tile_data)
                batch_positions.append(tile_info["position"])

            batch_inputs = torch.stack(batch_inputs).to(self.device)

            # Run inference
            with torch.no_grad():
                if mc_samples:
                    # MC Dropout inference for uncertainty
                    batch_probs, batch_uncertainty = self.model.predict_proba(
                        batch_inputs, mc_samples=mc_samples
                    )
                else:
                    # Standard inference
                    batch_logits = self.model(batch_inputs)
                    batch_probs = torch.sigmoid(batch_logits)
                    batch_uncertainty = None

            # Place predictions back into output arrays
            for j, (probs, pos) in enumerate(zip(batch_probs, batch_positions)):
                row_start, row_end, col_start, col_end = pos

                # Convert to numpy
                probs_np = probs.squeeze().cpu().numpy()

                # Create weight matrix for blending overlaps
                weight = self._create_tile_weight(self.tile_size, self.tile_overlap)

                # Accumulate predictions with weights
                prob_output[row_start:row_end, col_start:col_end] += probs_np * weight
                weight_output[row_start:row_end, col_start:col_end] += weight

                # Handle uncertainty if computed
                if mc_samples and batch_uncertainty is not None:
                    uncertainty_np = batch_uncertainty[j].squeeze().cpu().numpy()
                    uncertainty_output[row_start:row_end, col_start:col_end] += (
                        uncertainty_np * weight
                    )

            if (i + self.batch_size) % (self.batch_size * 10) == 0:
                logger.info(f"Processed {i + self.batch_size}/{len(tiles)} tiles")

        # Normalize by weights (handle overlaps)
        prob_output = np.divide(
            prob_output,
            weight_output,
            out=np.zeros_like(prob_output),
            where=weight_output != 0,
        )

        if mc_samples:
            uncertainty_output = np.divide(
                uncertainty_output,
                weight_output,
                out=np.zeros_like(uncertainty_output),
                where=weight_output != 0,
            )

        # Generate outputs
        output_paths = {}

        # 1. Probability GeoTIFF (flood_prob.tif)
        prob_path = output_dir / "flood_prob.tif"
        self._save_probability_geotiff(prob_output, input_data, prob_path)
        output_paths["probability"] = prob_path

        # 2. Binary extent GeoTIFF (flood_extent.tif)
        extent_binary = (prob_output >= threshold).astype(np.uint8)

        if apply_morphology:
            extent_binary = self._apply_morphology_cleaning(extent_binary)

        extent_path = output_dir / "flood_extent.tif"
        self._save_extent_geotiff(extent_binary, input_data, extent_path)
        output_paths["extent"] = extent_path

        # 3. Vector polygons (flood_extent.gpkg)
        vector_path = output_dir / "flood_extent.gpkg"
        self._polygonize_extent(extent_binary, input_data, vector_path)
        output_paths["vector"] = vector_path

        # 4. Uncertainty GeoTIFF (if computed)
        if mc_samples:
            uncertainty_path = output_dir / "flood_uncertainty.tif"
            self._save_probability_geotiff(
                uncertainty_output, input_data, uncertainty_path
            )
            output_paths["uncertainty"] = uncertainty_path

        # 5. Metadata
        metadata = self._generate_metadata(input_data, threshold, mc_samples)
        metadata_path = output_dir / "prediction_metadata.json"
        self._save_metadata(metadata, metadata_path)
        output_paths["metadata"] = metadata_path

        logger.info(f"Inference complete. Outputs saved to {output_dir}")
        return output_paths

    def _generate_inference_tiles(self, input_data: xr.Dataset) -> List[Dict]:
        """Generate tiles for inference with overlap."""
        height, width = input_data.dims["y"], input_data.dims["x"]
        tiles = []

        # Stack input channels
        channels = []
        for var in ["dem", "rainfall", "slope", "hand"]:  # Expected variables
            if var in input_data:
                channels.append(input_data[var].values)

        if len(channels) == 0:
            raise ValueError("No recognized input variables found in dataset")

        stacked_data = np.stack(channels, axis=0)  # [C, H, W]

        for i in range(0, height - self.tile_size + 1, self.stride):
            for j in range(0, width - self.tile_size + 1, self.stride):
                # Extract tile
                tile_data = stacked_data[
                    :, i : i + self.tile_size, j : j + self.tile_size
                ]

                # Normalize tile (per-tile normalization to prevent location leakage)
                tile_data = self._normalize_tile(tile_data)

                tiles.append(
                    {
                        "data": torch.tensor(tile_data, dtype=torch.float32),
                        "position": (i, i + self.tile_size, j, j + self.tile_size),
                    }
                )

        return tiles

    def _normalize_tile(self, tile_data: np.ndarray) -> np.ndarray:
        """Apply per-tile normalization as recommended in APPROACH.md."""
        normalized = np.zeros_like(tile_data)

        for c in range(tile_data.shape[0]):
            channel = tile_data[c]
            mean = np.mean(channel)
            std = np.std(channel)
            normalized[c] = (channel - mean) / (std + 1e-8)

        return normalized

    def _create_tile_weight(self, tile_size: int, overlap: int) -> np.ndarray:
        """Create weight matrix for blending overlapping tiles."""
        # Create distance from edges
        weight = np.ones((tile_size, tile_size), dtype=np.float32)

        if overlap > 0:
            # Taper weights near edges for smooth blending
            taper = overlap // 2
            for i in range(taper):
                alpha = (i + 1) / (taper + 1)
                weight[i, :] *= alpha
                weight[-(i + 1), :] *= alpha
                weight[:, i] *= alpha
                weight[:, -(i + 1)] *= alpha

        return weight

    def _apply_morphology_cleaning(self, binary_mask: np.ndarray) -> np.ndarray:
        """Apply morphological operations to clean flood extent."""
        # Remove small speckles (< N connected pixels)
        min_size = 10  # Minimum flood patch size

        # Label connected components
        labeled, num_features = ndimage.label(binary_mask)

        # Remove small components
        for label in range(1, num_features + 1):
            component_size = np.sum(labeled == label)
            if component_size < min_size:
                binary_mask[labeled == label] = 0

        # Close small gaps (1-2 pixel gaps)
        kernel = np.ones((3, 3), np.uint8)
        binary_mask = ndimage.binary_closing(binary_mask, structure=kernel).astype(
            np.uint8
        )

        return binary_mask

    def _save_probability_geotiff(
        self, prob_array: np.ndarray, reference_data: xr.Dataset, output_path: Path
    ):
        """Save probability array as GeoTIFF."""
        # Get spatial reference info
        crs = CRS.from_string(reference_data.attrs.get("crs", "EPSG:4326"))
        transform = reference_data.attrs.get("transform")

        if transform is None:
            # Create transform from coordinates
            x_coords = reference_data.coords["x"].values
            y_coords = reference_data.coords["y"].values

            transform = from_bounds(
                x_coords.min(),
                y_coords.min(),
                x_coords.max(),
                y_coords.max(),
                len(x_coords),
                len(y_coords),
            )

        # Save GeoTIFF
        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=prob_array.shape[0],
            width=prob_array.shape[1],
            count=1,
            dtype=rasterio.float32,
            crs=crs,
            transform=transform,
            compress="lzw",
        ) as dst:
            dst.write(prob_array, 1)
            dst.update_tags(
                description="Flood probability (0-1)",
                created_by="FloodRisk ML Pipeline",
            )

        logger.info(f"Saved probability GeoTIFF: {output_path}")

    def _save_extent_geotiff(
        self, extent_array: np.ndarray, reference_data: xr.Dataset, output_path: Path
    ):
        """Save binary extent as GeoTIFF."""
        crs = CRS.from_string(reference_data.attrs.get("crs", "EPSG:4326"))
        transform = reference_data.attrs.get("transform")

        if transform is None:
            x_coords = reference_data.coords["x"].values
            y_coords = reference_data.coords["y"].values

            transform = from_bounds(
                x_coords.min(),
                y_coords.min(),
                x_coords.max(),
                y_coords.max(),
                len(x_coords),
                len(y_coords),
            )

        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=extent_array.shape[0],
            width=extent_array.shape[1],
            count=1,
            dtype=rasterio.uint8,
            crs=crs,
            transform=transform,
            compress="lzw",
        ) as dst:
            dst.write(extent_array, 1)
            dst.update_tags(
                description="Binary flood extent (0=no flood, 1=flood)",
                created_by="FloodRisk ML Pipeline",
            )

        logger.info(f"Saved extent GeoTIFF: {output_path}")

    def _polygonize_extent(
        self, extent_array: np.ndarray, reference_data: xr.Dataset, output_path: Path
    ):
        """Convert binary extent to vector polygons."""
        crs = CRS.from_string(reference_data.attrs.get("crs", "EPSG:4326"))
        transform = reference_data.attrs.get("transform")

        if transform is None:
            x_coords = reference_data.coords["x"].values
            y_coords = reference_data.coords["y"].values

            transform = from_bounds(
                x_coords.min(),
                y_coords.min(),
                x_coords.max(),
                y_coords.max(),
                len(x_coords),
                len(y_coords),
            )

        # Extract polygons from raster
        shapes = features.shapes(
            extent_array.astype(np.uint8), mask=extent_array > 0, transform=transform
        )

        # Convert to GeoDataFrame
        geometries = []
        for geom, value in shapes:
            if value == 1:  # Flood areas
                geometries.append(shape(geom))

        if geometries:
            gdf = gpd.GeoDataFrame(
                {"flood_extent": [1] * len(geometries)}, geometry=geometries, crs=crs
            )

            # Dissolve overlapping polygons
            gdf_dissolved = gdf.dissolve(by="flood_extent")

            # Save to file
            gdf_dissolved.to_file(output_path, driver="GPKG")
            logger.info(f"Saved vector polygons: {output_path}")
        else:
            logger.warning("No flood polygons found to save")

    def _generate_metadata(
        self, input_data: xr.Dataset, threshold: float, mc_samples: Optional[int]
    ) -> Dict:
        """Generate prediction metadata."""
        metadata = {
            "model_info": {
                "architecture": self.model.__class__.__name__,
                "device": str(self.device),
                "parameters": sum(p.numel() for p in self.model.parameters()),
            },
            "prediction_settings": {
                "threshold": threshold,
                "tile_size": self.tile_size,
                "tile_overlap": self.tile_overlap,
                "batch_size": self.batch_size,
                "mc_samples": mc_samples,
            },
            "spatial_info": {
                "crs": input_data.attrs.get("crs"),
                "height": input_data.dims["y"],
                "width": input_data.dims["x"],
                "resolution": input_data.attrs.get("resolution"),
            },
            "input_variables": list(input_data.data_vars.keys()),
        }

        return metadata

    def _save_metadata(self, metadata: Dict, output_path: Path):
        """Save metadata as JSON."""
        import json

        with open(output_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"Saved metadata: {output_path}")


def batch_inference_from_config(
    model_path: Path, input_dir: Path, output_dir: Path, config: Dict
) -> List[Path]:
    """
    Run batch inference on multiple input datasets.

    Args:
        model_path: Path to trained model
        input_dir: Directory containing input datasets
        output_dir: Directory for outputs
        config: Inference configuration

    Returns:
        List of output directories created
    """
    logger.info(f"Starting batch inference from {input_dir}")

    # Load model
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path, map_location=device)

    # Initialize pipeline
    pipeline = GeoTIFFInferencePipeline(
        model=model,
        device=device,
        tile_size=config.get("tile_size", 512),
        tile_overlap=config.get("tile_overlap", 64),
        batch_size=config.get("batch_size", 4),
    )

    # Find input files
    input_files = list(input_dir.glob("*.nc"))  # Assuming NetCDF inputs
    if not input_files:
        input_files = list(input_dir.glob("*.tif"))  # Try GeoTIFF

    output_dirs = []

    for input_file in input_files:
        logger.info(f"Processing {input_file.name}")

        # Load input data
        if input_file.suffix == ".nc":
            input_data = xr.open_dataset(input_file)
        else:
            # Handle GeoTIFF input (would need additional processing)
            logger.warning(f"GeoTIFF input not fully implemented: {input_file}")
            continue

        # Create output directory
        output_subdir = output_dir / input_file.stem

        # Run inference
        try:
            outputs = pipeline.predict_large_raster(
                input_data=input_data,
                output_dir=output_subdir,
                threshold=config.get("threshold", 0.5),
                apply_morphology=config.get("apply_morphology", True),
                mc_samples=config.get("mc_samples", None),
            )

            output_dirs.append(output_subdir)
            logger.info(f"Completed inference for {input_file.name}")

        except Exception as e:
            logger.error(f"Failed inference for {input_file.name}: {e}")
            continue

    logger.info(f"Batch inference complete. Processed {len(output_dirs)} files")
    return output_dirs
