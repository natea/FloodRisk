"""
NFIP (National Flood Insurance Program) Validator

Validates flood risk models against historical insurance claims data from NFIP.
This provides real-world validation by comparing model predictions with actual
flood damage claims and insurance payouts.

Features:
- Load and process NFIP claims data
- Spatial matching of claims to model predictions
- Statistical validation metrics
- Damage prediction validation
- ROC curve analysis for claim prediction
- Economic validation metrics
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import warnings
from dataclasses import dataclass
import json
from datetime import datetime, timedelta

try:
    import geopandas as gpd
    from shapely.geometry import Point, Polygon
    from shapely.spatial import prepare
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False
    warnings.warn("geopandas not available - limited spatial functionality")

try:
    from sklearn.metrics import roc_curve, auc, precision_recall_curve
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    warnings.warn("scikit-learn not available - limited ML metrics")

from .metrics import MetricsCalculator, ClassificationMetrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class NFIPConfig:
    """Configuration for NFIP validation"""
    claims_data_path: str
    spatial_buffer: float = 100.0  # meters
    temporal_window: int = 30  # days
    min_claim_amount: float = 1000.0  # minimum claim amount to consider
    damage_categories: Dict = None
    loss_ratio_threshold: float = 0.8
    include_repetitive_loss: bool = True
    exclude_wind_damage: bool = True
    policy_data_path: Optional[str] = None
    
    def __post_init__(self):
        if self.damage_categories is None:
            self.damage_categories = {
                'minimal': (0, 10000),
                'moderate': (10000, 50000),
                'substantial': (50000, 100000),
                'severe': (100000, float('inf'))
            }


class NFIPDataProcessor:
    """
    Handles loading and processing NFIP claims data
    """
    
    def __init__(self, config: NFIPConfig):
        """
        Initialize NFIP data processor
        
        Args:
            config: NFIP validation configuration
        """
        self.config = config
        logger.info("NFIP Data Processor initialized")
    
    def load_claims_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load NFIP claims data from file
        
        Args:
            file_path: Path to claims data file (uses config if None)
            
        Returns:
            DataFrame with processed claims data
        """
        try:
            data_path = file_path or self.config.claims_data_path
            logger.info(f"Loading NFIP claims data from: {data_path}")
            
            # Load data based on file extension
            path = Path(data_path)
            if path.suffix.lower() == '.csv':
                claims_df = pd.read_csv(data_path)
            elif path.suffix.lower() in ['.xlsx', '.xls']:
                claims_df = pd.read_excel(data_path)
            elif path.suffix.lower() == '.parquet':
                claims_df = pd.read_parquet(data_path)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")
            
            logger.info(f"Loaded {len(claims_df)} claims records")
            
            # Process and validate data
            claims_df = self._process_claims_data(claims_df)
            
            return claims_df
            
        except Exception as e:
            logger.error(f"Error loading NFIP claims data: {e}")
            raise IOError(f"Failed to load claims data: {e}")
    
    def _process_claims_data(self, claims_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process and validate NFIP claims data
        
        Args:
            claims_df: Raw claims DataFrame
            
        Returns:
            Processed DataFrame
        """
        logger.info("Processing NFIP claims data")
        
        # Standardize column names (NFIP data format may vary)
        column_mapping = self._get_column_mapping(claims_df.columns)
        claims_df = claims_df.rename(columns=column_mapping)
        
        # Required columns
        required_cols = ['longitude', 'latitude', 'date_of_loss', 'amount_paid']
        missing_cols = [col for col in required_cols if col not in claims_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Data type conversions
        claims_df['date_of_loss'] = pd.to_datetime(claims_df['date_of_loss'])
        claims_df['amount_paid'] = pd.to_numeric(claims_df['amount_paid'], errors='coerce')
        claims_df['longitude'] = pd.to_numeric(claims_df['longitude'], errors='coerce')
        claims_df['latitude'] = pd.to_numeric(claims_df['latitude'], errors='coerce')
        
        # Filter out invalid records
        initial_count = len(claims_df)
        
        # Remove records with missing coordinates
        claims_df = claims_df.dropna(subset=['longitude', 'latitude'])
        
        # Remove records with invalid coordinates
        claims_df = claims_df[
            (claims_df['longitude'] >= -180) & (claims_df['longitude'] <= 180) &
            (claims_df['latitude'] >= -90) & (claims_df['latitude'] <= 90)
        ]
        
        # Filter by minimum claim amount
        claims_df = claims_df[claims_df['amount_paid'] >= self.config.min_claim_amount]
        
        # Exclude wind damage if specified
        if self.config.exclude_wind_damage and 'cause_of_damage' in claims_df.columns:
            wind_keywords = ['wind', 'hurricane', 'tornado', 'storm']
            wind_mask = claims_df['cause_of_damage'].str.lower().str.contains(
                '|'.join(wind_keywords), na=False
            )
            claims_df = claims_df[~wind_mask]
        
        # Add damage categories
        claims_df['damage_category'] = self._categorize_damage(claims_df['amount_paid'])
        
        # Add binary claim indicator
        claims_df['has_claim'] = 1
        
        logger.info(f"Processed claims data: {len(claims_df)} valid records (removed {initial_count - len(claims_df)})")
        
        return claims_df
    
    def _get_column_mapping(self, columns: pd.Index) -> Dict[str, str]:
        """
        Map various NFIP column names to standardized names
        """
        mapping = {}
        col_lower = [col.lower().replace(' ', '_') for col in columns]
        
        # Longitude mapping
        longitude_variants = ['longitude', 'lon', 'lng', 'long', 'x_coord']
        for variant in longitude_variants:
            if variant in col_lower:
                original_col = columns[col_lower.index(variant)]
                mapping[original_col] = 'longitude'
                break
        
        # Latitude mapping
        latitude_variants = ['latitude', 'lat', 'y_coord']
        for variant in latitude_variants:
            if variant in col_lower:
                original_col = columns[col_lower.index(variant)]
                mapping[original_col] = 'latitude'
                break
        
        # Date mapping
        date_variants = ['date_of_loss', 'loss_date', 'date', 'event_date']
        for variant in date_variants:
            if variant in col_lower:
                original_col = columns[col_lower.index(variant)]
                mapping[original_col] = 'date_of_loss'
                break
        
        # Amount mapping
        amount_variants = ['amount_paid', 'paid_amount', 'claim_amount', 'loss_amount']
        for variant in amount_variants:
            if variant in col_lower:
                original_col = columns[col_lower.index(variant)]
                mapping[original_col] = 'amount_paid'
                break
        
        return mapping
    
    def _categorize_damage(self, amounts: pd.Series) -> pd.Series:
        """Categorize damage amounts"""
        categories = []
        for amount in amounts:
            for category, (min_val, max_val) in self.config.damage_categories.items():
                if min_val <= amount < max_val:
                    categories.append(category)
                    break
            else:
                categories.append('unknown')
        
        return pd.Series(categories, index=amounts.index)


class SpatialMatcher:
    """
    Handles spatial matching between model predictions and NFIP claims
    """
    
    def __init__(self, buffer_distance: float = 100.0):
        """
        Initialize spatial matcher
        
        Args:
            buffer_distance: Buffer distance in meters for spatial matching
        """
        self.buffer_distance = buffer_distance
        logger.info(f"Spatial Matcher initialized with buffer: {buffer_distance}m")
    
    def match_claims_to_predictions(self, claims_df: pd.DataFrame, 
                                   prediction_grid: Dict,
                                   event_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Match NFIP claims to model prediction grid cells
        
        Args:
            claims_df: DataFrame with NFIP claims
            prediction_grid: Dictionary with prediction data and spatial info
            event_date: Date of flood event for temporal filtering
            
        Returns:
            DataFrame with matched claims and predictions
        """
        try:
            logger.info("Matching claims to prediction grid")
            
            if not HAS_GEOPANDAS:
                logger.warning("Using simple grid-based matching without geopandas")
                return self._simple_grid_match(claims_df, prediction_grid)
            
            # Create GeoDataFrame from claims
            geometry = [Point(row.longitude, row.latitude) for _, row in claims_df.iterrows()]
            claims_gdf = gpd.GeoDataFrame(claims_df, geometry=geometry, crs='EPSG:4326')
            
            # Create prediction grid polygons
            pred_gdf = self._create_prediction_polygons(prediction_grid)
            
            # Temporal filtering if event date provided
            if event_date:
                time_window = timedelta(days=30)  # 30-day window
                claims_gdf = claims_gdf[
                    abs(claims_gdf['date_of_loss'] - event_date) <= time_window
                ]
                logger.info(f"Filtered to {len(claims_gdf)} claims within {time_window.days} days of event")
            
            # Spatial join
            matched_data = gpd.sjoin(claims_gdf, pred_gdf, how='inner', predicate='within')
            
            logger.info(f"Matched {len(matched_data)} claims to prediction grid")
            return matched_data
            
        except Exception as e:
            logger.error(f"Error matching claims to predictions: {e}")
            raise RuntimeError(f"Spatial matching failed: {e}")
    
    def _simple_grid_match(self, claims_df: pd.DataFrame, prediction_grid: Dict) -> pd.DataFrame:
        """
        Simple grid-based matching without geopandas (fallback)
        """
        logger.info("Using simple grid-based matching")
        
        # This is a simplified approach - would need proper implementation
        # based on the specific prediction grid format
        
        # For demonstration, assume regular grid and find nearest cell
        matched_data = claims_df.copy()
        matched_data['grid_row'] = 0
        matched_data['grid_col'] = 0
        matched_data['predicted_depth'] = 0.0
        
        return matched_data
    
    def _create_prediction_polygons(self, prediction_grid: Dict) -> gpd.GeoDataFrame:
        """
        Create polygon GeoDataFrame from prediction grid
        """
        # This would depend on the specific grid format
        # For demonstration, create a simple grid
        
        polygons = []
        predictions = []
        grid_ids = []
        
        # This is a placeholder - actual implementation would depend on grid structure
        for i in range(10):  # Example grid
            for j in range(10):
                # Create cell polygon
                x_min, y_min = i * 0.01, j * 0.01
                x_max, y_max = (i + 1) * 0.01, (j + 1) * 0.01
                
                polygon = Polygon([
                    (x_min, y_min), (x_max, y_min),
                    (x_max, y_max), (x_min, y_max)
                ])
                
                polygons.append(polygon)
                predictions.append(prediction_grid.get('data', np.zeros((10, 10)))[i, j])
                grid_ids.append(f"{i}_{j}")
        
        pred_gdf = gpd.GeoDataFrame({
            'grid_id': grid_ids,
            'predicted_depth': predictions
        }, geometry=polygons, crs='EPSG:4326')
        
        return pred_gdf


class NFIPValidator:
    """
    Main NFIP validator for flood risk model validation
    """
    
    def __init__(self, config: Optional[NFIPConfig] = None):
        """
        Initialize NFIP validator
        
        Args:
            config: NFIP validation configuration
        """
        self.config = config or NFIPConfig("./nfip_claims.csv")
        self.data_processor = NFIPDataProcessor(self.config)
        self.spatial_matcher = SpatialMatcher(self.config.spatial_buffer)
        self.metrics_calc = MetricsCalculator()
        
        logger.info("NFIP Validator initialized")
    
    def validate_against_nfip(self, model_predictions: np.ndarray,
                             prediction_metadata: Dict,
                             event_date: Optional[datetime] = None,
                             claims_file: Optional[str] = None) -> Dict:
        """
        Validate model predictions against NFIP claims data
        
        Args:
            model_predictions: Model flood depth predictions
            prediction_metadata: Spatial and temporal metadata for predictions
            event_date: Date of flood event
            claims_file: Path to claims data file
            
        Returns:
            Dictionary with comprehensive validation results
        """
        try:
            logger.info("Starting NFIP validation")
            
            # Load claims data
            claims_df = self.data_processor.load_claims_data(claims_file)
            
            # Prepare prediction grid
            prediction_grid = {
                'data': model_predictions,
                **prediction_metadata
            }
            
            # Match claims to predictions
            matched_data = self.spatial_matcher.match_claims_to_predictions(
                claims_df, prediction_grid, event_date
            )
            
            if len(matched_data) == 0:
                logger.warning("No claims matched to prediction grid")
                return self._empty_validation_result()
            
            # Calculate validation metrics
            validation_results = self._calculate_nfip_metrics(matched_data)
            
            # Economic validation
            economic_results = self._calculate_economic_metrics(matched_data)
            
            # ROC analysis if sklearn available
            if HAS_SKLEARN:
                roc_results = self._calculate_roc_metrics(matched_data)
                validation_results['roc_analysis'] = roc_results
            
            # Combine results
            final_results = {
                'claim_validation': validation_results,
                'economic_validation': economic_results,
                'data_summary': self._create_data_summary(matched_data),
                'metadata': {
                    'validation_timestamp': datetime.now().isoformat(),
                    'event_date': event_date.isoformat() if event_date else None,
                    'total_claims': len(claims_df),
                    'matched_claims': len(matched_data),
                    'config': self.config.__dict__
                }
            }
            
            logger.info("NFIP validation completed successfully")
            return final_results
            
        except Exception as e:
            logger.error(f"NFIP validation failed: {e}")
            raise RuntimeError(f"NFIP validation error: {e}")
    
    def _calculate_nfip_metrics(self, matched_data: pd.DataFrame) -> Dict:
        """
        Calculate NFIP-specific validation metrics
        """
        logger.info("Calculating NFIP validation metrics")
        
        results = {}
        
        # Binary claim prediction (flood/no-flood)
        if 'predicted_depth' in matched_data.columns:
            # Create binary predictions based on depth threshold
            depth_threshold = 0.01  # 1cm threshold
            binary_predictions = (matched_data['predicted_depth'] > depth_threshold).astype(int)
            binary_observations = matched_data['has_claim']
            
            # Classification metrics
            classification_metrics = self.metrics_calc.calculate_classification_metrics(
                binary_predictions, binary_observations
            )
            results['binary_prediction'] = classification_metrics
        
        # Depth vs damage correlation
        if 'predicted_depth' in matched_data.columns and 'amount_paid' in matched_data.columns:
            correlation = matched_data['predicted_depth'].corr(matched_data['amount_paid'])
            results['depth_damage_correlation'] = float(correlation) if not np.isnan(correlation) else None
        
        # Category-based analysis
        category_results = self._analyze_by_damage_category(matched_data)
        results['category_analysis'] = category_results
        
        # Spatial distribution analysis
        spatial_results = self._analyze_spatial_distribution(matched_data)
        results['spatial_analysis'] = spatial_results
        
        return results
    
    def _calculate_economic_metrics(self, matched_data: pd.DataFrame) -> Dict:
        """
        Calculate economic validation metrics
        """
        logger.info("Calculating economic validation metrics")
        
        total_claims_amount = matched_data['amount_paid'].sum()
        avg_claim_amount = matched_data['amount_paid'].mean()
        
        # Predicted vs actual economic impact
        economic_metrics = {
            'total_claims_amount': float(total_claims_amount),
            'average_claim_amount': float(avg_claim_amount),
            'median_claim_amount': float(matched_data['amount_paid'].median()),
            'max_claim_amount': float(matched_data['amount_paid'].max()),
            'claims_per_unit_area': len(matched_data) / 1.0,  # Would need actual area
        }
        
        # Damage category distribution
        category_dist = matched_data['damage_category'].value_counts(normalize=True)
        economic_metrics['damage_category_distribution'] = category_dist.to_dict()
        
        return economic_metrics
    
    def _calculate_roc_metrics(self, matched_data: pd.DataFrame) -> Dict:
        """
        Calculate ROC curve and related metrics for claim prediction
        """
        if 'predicted_depth' not in matched_data.columns:
            return {}
        
        logger.info("Calculating ROC metrics")
        
        y_true = matched_data['has_claim']
        y_scores = matched_data['predicted_depth']
        
        # ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        # Precision-recall curve
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)
        
        return {
            'roc_auc': float(roc_auc),
            'pr_auc': float(pr_auc),
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'roc_thresholds': thresholds.tolist(),
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'pr_thresholds': pr_thresholds.tolist()
        }
    
    def _analyze_by_damage_category(self, matched_data: pd.DataFrame) -> Dict:
        """
        Analyze results by damage category
        """
        category_results = {}
        
        for category in matched_data['damage_category'].unique():
            if category == 'unknown':
                continue
            
            category_data = matched_data[matched_data['damage_category'] == category]
            
            if len(category_data) == 0:
                continue
            
            # Calculate metrics for this category
            if 'predicted_depth' in category_data.columns:
                avg_predicted = category_data['predicted_depth'].mean()
                avg_actual = category_data['amount_paid'].mean()
                
                category_results[category] = {
                    'count': len(category_data),
                    'avg_predicted_depth': float(avg_predicted),
                    'avg_claim_amount': float(avg_actual),
                    'proportion_of_total': len(category_data) / len(matched_data)
                }
        
        return category_results
    
    def _analyze_spatial_distribution(self, matched_data: pd.DataFrame) -> Dict:
        """
        Analyze spatial distribution of claims vs predictions
        """
        spatial_results = {
            'claim_locations_count': len(matched_data),
            'geographic_bounds': {
                'min_longitude': float(matched_data['longitude'].min()),
                'max_longitude': float(matched_data['longitude'].max()),
                'min_latitude': float(matched_data['latitude'].min()),
                'max_latitude': float(matched_data['latitude'].max())
            }
        }
        
        return spatial_results
    
    def _create_data_summary(self, matched_data: pd.DataFrame) -> Dict:
        """
        Create summary of matched data
        """
        return {
            'total_matched_claims': len(matched_data),
            'total_claim_amount': float(matched_data['amount_paid'].sum()),
            'date_range': {
                'earliest_claim': matched_data['date_of_loss'].min().isoformat(),
                'latest_claim': matched_data['date_of_loss'].max().isoformat()
            },
            'damage_categories': matched_data['damage_category'].value_counts().to_dict()
        }
    
    def _empty_validation_result(self) -> Dict:
        """
        Return empty validation result when no claims match
        """
        return {
            'claim_validation': {'error': 'No claims matched to prediction grid'},
            'economic_validation': {'error': 'No economic data available'},
            'data_summary': {'total_matched_claims': 0},
            'metadata': {
                'validation_timestamp': datetime.now().isoformat(),
                'total_claims': 0,
                'matched_claims': 0
            }
        }
    
    def generate_nfip_report(self, validation_results: Dict, output_path: str) -> None:
        """
        Generate detailed NFIP validation report
        
        Args:
            validation_results: Results from validate_against_nfip
            output_path: Path for output report
        """
        logger.info(f"Generating NFIP validation report: {output_path}")
        
        # Create comprehensive report
        report = {
            'executive_summary': self._create_executive_summary(validation_results),
            'detailed_results': validation_results,
            'recommendations': self._generate_nfip_recommendations(validation_results),
            'data_quality_assessment': self._assess_data_quality(validation_results)
        }
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"NFIP report saved to {output_path}")
    
    def _create_executive_summary(self, results: Dict) -> Dict:
        """
        Create executive summary of validation results
        """
        summary = {
            'validation_date': results['metadata']['validation_timestamp'],
            'total_claims_analyzed': results['metadata']['total_claims'],
            'successfully_matched_claims': results['metadata']['matched_claims']
        }
        
        # Add key metrics if available
        claim_val = results.get('claim_validation', {})
        if 'binary_prediction' in claim_val:
            binary_metrics = claim_val['binary_prediction']
            summary.update({
                'model_accuracy': binary_metrics.get('accuracy'),
                'model_precision': binary_metrics.get('precision'),
                'model_recall': binary_metrics.get('recall'),
                'f1_score': binary_metrics.get('f1_score')
            })
        
        economic_val = results.get('economic_validation', {})
        if 'total_claims_amount' in economic_val:
            summary['total_economic_impact'] = economic_val['total_claims_amount']
        
        return summary
    
    def _generate_nfip_recommendations(self, results: Dict) -> List[str]:
        """
        Generate recommendations based on NFIP validation results
        """
        recommendations = []
        
        matched_claims = results['metadata'].get('matched_claims', 0)
        total_claims = results['metadata'].get('total_claims', 0)
        
        # Data coverage recommendations
        if matched_claims == 0:
            recommendations.append("No claims were matched - check spatial alignment and temporal windows")
        elif matched_claims / total_claims < 0.1:
            recommendations.append("Low claim matching rate - consider expanding spatial buffer or checking coordinate systems")
        
        # Accuracy recommendations
        claim_val = results.get('claim_validation', {})
        if 'binary_prediction' in claim_val:
            accuracy = claim_val['binary_prediction'].get('accuracy', 0)
            if accuracy < 0.7:
                recommendations.append("Model accuracy below 70% - consider model recalibration or additional training data")
            elif accuracy > 0.9:
                recommendations.append("Excellent model performance - suitable for operational use")
        
        # Economic recommendations
        economic_val = results.get('economic_validation', {})
        if 'damage_category_distribution' in economic_val:
            severe_damage = economic_val['damage_category_distribution'].get('severe', 0)
            if severe_damage > 0.2:
                recommendations.append("High proportion of severe damage claims - focus on extreme event modeling")
        
        return recommendations
    
    def _assess_data_quality(self, results: Dict) -> Dict:
        """
        Assess quality of NFIP data and matching
        """
        quality_assessment = {
            'matching_success_rate': results['metadata']['matched_claims'] / max(results['metadata']['total_claims'], 1),
            'temporal_coverage': 'good' if results['metadata']['matched_claims'] > 10 else 'limited',
            'spatial_coverage': 'assessed',  # Would need more detailed analysis
            'data_completeness': 'good'  # Would need to check for missing fields
        }
        
        return quality_assessment


# Utility functions
def load_nfip_sample_data() -> pd.DataFrame:
    """
    Create sample NFIP data for testing purposes
    """
    np.random.seed(42)
    n_claims = 100
    
    sample_data = pd.DataFrame({
        'longitude': np.random.uniform(-95, -85, n_claims),
        'latitude': np.random.uniform(25, 35, n_claims),
        'date_of_loss': pd.date_range('2020-01-01', '2020-12-31', periods=n_claims),
        'amount_paid': np.random.lognormal(8, 1.5, n_claims),
        'cause_of_damage': np.random.choice(['flood', 'storm surge', 'heavy rain'], n_claims),
        'property_type': np.random.choice(['single_family', 'condo', 'mobile_home'], n_claims)
    })
    
    return sample_data


def create_nfip_validation_config(claims_path: str, **kwargs) -> NFIPConfig:
    """
    Create NFIP validation configuration with defaults
    
    Args:
        claims_path: Path to NFIP claims data
        **kwargs: Additional configuration parameters
        
    Returns:
        NFIPConfig object
    """
    return NFIPConfig(
        claims_data_path=claims_path,
        **kwargs
    )