#!/usr/bin/env python3
"""
Check the status of available data for training the flood prediction model.
"""

import argparse
from pathlib import Path
import json
import logging
from typing import Dict, List, Optional
import numpy as np
import rasterio
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
console = Console()


def check_file_exists(path: Path) -> Dict:
    """Check if a file exists and get its properties."""
    if not path.exists():
        return {'exists': False}
    
    stats = path.stat()
    info = {
        'exists': True,
        'size_mb': stats.st_size / (1024 * 1024),
        'modified': stats.st_mtime
    }
    
    # Try to get raster properties if it's a GeoTIFF
    if path.suffix.lower() in ['.tif', '.tiff']:
        try:
            with rasterio.open(path) as src:
                info.update({
                    'width': src.width,
                    'height': src.height,
                    'bands': src.count,
                    'dtype': str(src.dtypes[0]),
                    'crs': str(src.crs),
                    'resolution': src.res
                })
        except Exception as e:
            logger.debug(f"Could not read raster properties: {e}")
    
    return info


def check_directory(path: Path, pattern: str = "*") -> Dict:
    """Check directory contents."""
    if not path.exists():
        return {'exists': False, 'files': []}
    
    files = list(path.glob(pattern))
    total_size = sum(f.stat().st_size for f in files if f.is_file()) / (1024 * 1024)
    
    return {
        'exists': True,
        'n_files': len(files),
        'total_size_mb': total_size,
        'files': [f.name for f in files[:5]]  # First 5 files
    }


def check_dem_data(data_dir: Path) -> Dict:
    """Check DEM data availability."""
    dem_status = {}
    
    # Check for Nashville DEM
    nashville_paths = [
        data_dir / 'nashville' / 'test_dem.npy',
        data_dir / 'regions' / 'nashville' / 'dem' / 'nashville_dem_10m.tif',
        data_dir / 'regions' / 'nashville' / 'dem' / 'nashville_dem_10m_1.tif',
    ]
    
    for path in nashville_paths:
        if path.exists():
            dem_status['nashville_dem'] = check_file_exists(path)
            dem_status['nashville_dem']['path'] = str(path)
            break
    else:
        dem_status['nashville_dem'] = {'exists': False}
    
    # Check for DEM tiles
    dem_tiles_dir = data_dir / 'regions' / 'nashville' / 'dem'
    if dem_tiles_dir.exists():
        tiles = list(dem_tiles_dir.glob("*.tif"))
        dem_status['dem_tiles'] = {
            'exists': True,
            'n_tiles': len(tiles),
            'total_size_mb': sum(t.stat().st_size for t in tiles) / (1024 * 1024)
        }
    else:
        dem_status['dem_tiles'] = {'exists': False}
    
    return dem_status


def check_precipitation_data(data_dir: Path) -> Dict:
    """Check precipitation data availability."""
    precip_status = {}
    
    # Check for precipitation grids
    precip_dirs = [
        data_dir / 'v2_additional' / 'precipitation_grids' / 'processed',
        data_dir / 'regions' / 'nashville' / 'rainfall',
    ]
    
    for precip_dir in precip_dirs:
        if precip_dir.exists():
            precip_status[precip_dir.name] = check_directory(precip_dir, "*.csv")
            precip_status[precip_dir.name]['path'] = str(precip_dir)
    
    return precip_status


def check_training_tiles(data_dir: Path) -> Dict:
    """Check training tiles."""
    tiles_status = {}
    tiles_dir = data_dir / 'training_tiles'
    
    if not tiles_dir.exists():
        return {'exists': False}
    
    for split in ['train', 'val', 'test']:
        split_dir = tiles_dir / split
        if split_dir.exists():
            tiles_status[split] = {
                'dem': check_directory(split_dir / 'dem', "*.tif"),
                'precipitation': check_directory(split_dir / 'precipitation', "*.tif"),
                'metadata': check_file_exists(split_dir / 'metadata.json')
            }
    
    # Check tiling config
    config_path = tiles_dir / 'tiling_config.json'
    if config_path.exists():
        with open(config_path) as f:
            tiles_status['config'] = json.load(f)
    
    return tiles_status


def check_simulation_results(results_dir: Path) -> Dict:
    """Check LISFLOOD-FP simulation results."""
    if not results_dir.exists():
        return {'exists': False}
    
    sim_dirs = list(results_dir.glob("*"))
    
    return {
        'exists': True,
        'n_simulations': len(sim_dirs),
        'simulations': [d.name for d in sim_dirs[:5]]
    }


def check_model_outputs(outputs_dir: Path) -> Dict:
    """Check trained model outputs."""
    model_status = {}
    
    if not outputs_dir.exists():
        return {'exists': False}
    
    # Check for baseline model
    baseline_dir = outputs_dir / 'baseline'
    if baseline_dir.exists():
        model_files = list(baseline_dir.glob("*.pth")) + list(baseline_dir.glob("*.ckpt"))
        model_status['baseline'] = {
            'exists': True,
            'n_models': len(model_files),
            'models': [f.name for f in model_files]
        }
    
    return model_status


def print_status_report(status: Dict):
    """Print a formatted status report."""
    
    # DEM Data
    console.print("\n[bold cyan]DEM Data Status[/bold cyan]")
    table = Table(box=box.ROUNDED)
    table.add_column("Component", style="yellow")
    table.add_column("Status", style="green")
    table.add_column("Details")
    
    dem_status = status.get('dem', {})
    if dem_status.get('nashville_dem', {}).get('exists'):
        dem_info = dem_status['nashville_dem']
        table.add_row(
            "Nashville DEM",
            "✅ Available",
            f"Size: {dem_info.get('size_mb', 0):.1f} MB, "
            f"Resolution: {dem_info.get('resolution', 'Unknown')}"
        )
    else:
        table.add_row("Nashville DEM", "❌ Missing", "Run download_nashville_data.py")
    
    if dem_status.get('dem_tiles', {}).get('exists'):
        tiles_info = dem_status['dem_tiles']
        table.add_row(
            "DEM Tiles",
            "✅ Available",
            f"{tiles_info['n_tiles']} tiles, {tiles_info['total_size_mb']:.1f} MB total"
        )
    else:
        table.add_row("DEM Tiles", "❌ Missing", "")
    
    console.print(table)
    
    # Precipitation Data
    console.print("\n[bold cyan]Precipitation Data Status[/bold cyan]")
    table = Table(box=box.ROUNDED)
    table.add_column("Source", style="yellow")
    table.add_column("Status", style="green")
    table.add_column("Files")
    
    precip_status = status.get('precipitation', {})
    for source, info in precip_status.items():
        if info.get('exists'):
            table.add_row(
                source,
                "✅ Available",
                f"{info['n_files']} files, {info['total_size_mb']:.1f} MB"
            )
    
    if not precip_status or not any(p.get('exists') for p in precip_status.values()):
        table.add_row("Precipitation", "❌ Missing", "Generate or download precipitation data")
    
    console.print(table)
    
    # Training Tiles
    console.print("\n[bold cyan]Training Tiles Status[/bold cyan]")
    tiles_status = status.get('training_tiles', {})
    
    if tiles_status.get('exists') == False:
        console.print("[red]❌ No training tiles found. Run create_training_tiles.py[/red]")
    else:
        table = Table(box=box.ROUNDED)
        table.add_column("Split", style="yellow")
        table.add_column("DEM Tiles")
        table.add_column("Precip Tiles")
        
        for split in ['train', 'val', 'test']:
            if split in tiles_status:
                split_info = tiles_status[split]
                table.add_row(
                    split.capitalize(),
                    f"{split_info.get('dem', {}).get('n_files', 0)} tiles",
                    f"{split_info.get('precipitation', {}).get('n_files', 0)} tiles"
                )
        
        console.print(table)
        
        if 'config' in tiles_status:
            config = tiles_status['config']
            console.print(f"\n[dim]Tile size: {config.get('tile_size')}x{config.get('tile_size')}, "
                         f"Overlap: {config.get('overlap')} pixels[/dim]")
    
    # Simulations
    console.print("\n[bold cyan]Simulation Results[/bold cyan]")
    sim_status = status.get('simulations', {})
    if sim_status.get('exists'):
        console.print(f"✅ {sim_status['n_simulations']} simulations found")
        if sim_status['simulations']:
            console.print(f"[dim]Recent: {', '.join(sim_status['simulations'])}[/dim]")
    else:
        console.print("[yellow]⚠️ No simulation results found[/yellow]")
    
    # Model Outputs
    console.print("\n[bold cyan]Trained Models[/bold cyan]")
    model_status = status.get('models', {})
    if model_status.get('baseline', {}).get('exists'):
        baseline = model_status['baseline']
        console.print(f"✅ Baseline model: {baseline['n_models']} checkpoints")
        if baseline['models']:
            console.print(f"[dim]Files: {', '.join(baseline['models'])}[/dim]")
    else:
        console.print("[yellow]⚠️ No trained models found[/yellow]")
    
    # Summary
    console.print("\n" + "="*50)
    ready_components = []
    missing_components = []
    
    if dem_status.get('nashville_dem', {}).get('exists'):
        ready_components.append("DEM")
    else:
        missing_components.append("DEM")
    
    if any(p.get('exists') for p in precip_status.values()):
        ready_components.append("Precipitation")
    else:
        missing_components.append("Precipitation")
    
    if tiles_status and tiles_status != {'exists': False}:
        ready_components.append("Training tiles")
    else:
        missing_components.append("Training tiles")
    
    if ready_components:
        console.print(f"[green]✅ Ready: {', '.join(ready_components)}[/green]")
    if missing_components:
        console.print(f"[red]❌ Missing: {', '.join(missing_components)}[/red]")
    
    # Next steps
    if missing_components:
        console.print("\n[bold yellow]Next Steps:[/bold yellow]")
        if "DEM" in missing_components:
            console.print("1. Run: python scripts/data_acquisition/download_nashville_data.py")
        if "Precipitation" in missing_components:
            console.print("2. Run: python scripts/data_processing/generate_nashville_precipitation.py")
        if "Training tiles" in missing_components and "DEM" not in missing_components:
            console.print("3. Run: python scripts/create_training_tiles.py")


def main():
    parser = argparse.ArgumentParser(description="Check data status for flood model training")
    parser.add_argument('--data-dir', type=Path, default=Path('data'),
                       help='Base data directory (default: data)')
    parser.add_argument('--results-dir', type=Path, default=Path('results'),
                       help='Results directory (default: results)')
    parser.add_argument('--outputs-dir', type=Path, default=Path('outputs'),
                       help='Model outputs directory (default: outputs)')
    parser.add_argument('--json', action='store_true',
                       help='Output as JSON instead of formatted text')
    
    args = parser.parse_args()
    
    # Collect status information
    status = {
        'dem': check_dem_data(args.data_dir),
        'precipitation': check_precipitation_data(args.data_dir),
        'training_tiles': check_training_tiles(args.data_dir),
        'simulations': check_simulation_results(args.results_dir / 'simulations'),
        'models': check_model_outputs(args.outputs_dir)
    }
    
    if args.json:
        # Output as JSON
        print(json.dumps(status, indent=2, default=str))
    else:
        # Print formatted report
        print_status_report(status)


if __name__ == "__main__":
    main()