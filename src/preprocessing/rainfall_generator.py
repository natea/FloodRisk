"""Rainfall scenario generation module.

This module provides functionality for generating rainfall scenarios based on
NOAA Atlas 14 precipitation frequency data and synthetic storm generation.
"""

import numpy as np
import pandas as pd
from scipy import stats, interpolate
from typing import Dict, List, Tuple, Optional, Union
import warnings
from pathlib import Path
import json


class RainfallGenerator:
    """Generates rainfall scenarios for flood risk analysis.
    
    This class provides methods for creating synthetic rainfall events based on
    statistical distributions, NOAA Atlas 14 data, and design storm patterns.
    """
    
    def __init__(self, time_step_minutes: int = 15):
        """Initialize rainfall generator.
        
        Args:
            time_step_minutes: Time step for rainfall data in minutes
        """
        self.time_step_minutes = time_step_minutes
        self.time_step_hours = time_step_minutes / 60.0
        
        # Standard return periods for NOAA Atlas 14
        self.return_periods = [1, 2, 5, 10, 25, 50, 100, 200, 500, 1000]
        
        # Standard durations in hours
        self.durations = [1, 2, 3, 6, 12, 24, 48]
        
        # Design storm distributions (normalized)
        self.storm_distributions = {
            'scs_type_i': self._get_scs_type_i(),
            'scs_type_ia': self._get_scs_type_ia(),
            'scs_type_ii': self._get_scs_type_ii(),
            'scs_type_iii': self._get_scs_type_iii(),
            'uniform': self._get_uniform_distribution(),
            'chicago': self._get_chicago_distribution()
        }
    
    def _get_scs_type_i(self) -> np.ndarray:
        """Get SCS Type I rainfall distribution (Pacific maritime climate).
        
        Returns:
            Normalized cumulative rainfall distribution
        """
        # Time fractions for 24-hour storm
        time_fractions = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
                                  0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])
        
        # Cumulative rainfall fractions
        cumulative_fractions = np.array([0, 0.08, 0.17, 0.26, 0.35, 0.44, 0.52, 0.60, 0.67, 0.73,
                                        0.79, 0.84, 0.88, 0.91, 0.94, 0.96, 0.98, 0.99, 0.995, 0.999, 1.0])
        
        return np.column_stack([time_fractions, cumulative_fractions])
    
    def _get_scs_type_ia(self) -> np.ndarray:
        """Get SCS Type IA rainfall distribution (Pacific maritime climate, 24-hour).
        
        Returns:
            Normalized cumulative rainfall distribution
        """
        time_fractions = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
                                  0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])
        
        cumulative_fractions = np.array([0, 0.05, 0.11, 0.19, 0.28, 0.39, 0.50, 0.62, 0.72, 0.79,
                                        0.85, 0.89, 0.92, 0.94, 0.96, 0.97, 0.98, 0.99, 0.995, 0.999, 1.0])
        
        return np.column_stack([time_fractions, cumulative_fractions])
    
    def _get_scs_type_ii(self) -> np.ndarray:
        """Get SCS Type II rainfall distribution (temperate continental climate).
        
        Returns:
            Normalized cumulative rainfall distribution
        """
        time_fractions = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
                                  0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])
        
        cumulative_fractions = np.array([0, 0.02, 0.05, 0.08, 0.12, 0.16, 0.22, 0.28, 0.36, 0.46,
                                        0.66, 0.76, 0.82, 0.86, 0.89, 0.92, 0.94, 0.96, 0.98, 0.99, 1.0])
        
        return np.column_stack([time_fractions, cumulative_fractions])
    
    def _get_scs_type_iii(self) -> np.ndarray:
        """Get SCS Type III rainfall distribution (subtropical climate).
        
        Returns:
            Normalized cumulative rainfall distribution
        """
        time_fractions = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
                                  0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])
        
        cumulative_fractions = np.array([0, 0.02, 0.04, 0.06, 0.08, 0.11, 0.15, 0.20, 0.27, 0.36,
                                        0.50, 0.62, 0.71, 0.78, 0.84, 0.88, 0.92, 0.95, 0.97, 0.99, 1.0])
        
        return np.column_stack([time_fractions, cumulative_fractions])
    
    def _get_uniform_distribution(self) -> np.ndarray:
        """Get uniform rainfall distribution.
        
        Returns:
            Normalized cumulative rainfall distribution
        """
        time_fractions = np.linspace(0, 1, 21)
        cumulative_fractions = time_fractions.copy()
        
        return np.column_stack([time_fractions, cumulative_fractions])
    
    def _get_chicago_distribution(self, a: float = 0.1, b: float = 0.8) -> np.ndarray:
        """Get Chicago design storm distribution.
        
        Args:
            a: Storm parameter (dimensionless)
            b: Storm parameter (dimensionless)
            
        Returns:
            Normalized cumulative rainfall distribution
        """
        time_fractions = np.linspace(0, 1, 100)
        
        # Chicago storm intensity equation
        intensities = np.zeros_like(time_fractions)
        
        # Before peak (t < tb)
        tb = 0.3  # Time to peak (fraction of total duration)
        
        for i, t in enumerate(time_fractions):
            if t <= tb:
                intensities[i] = a * (tb - t)**(-b)
            else:
                intensities[i] = a * (t - tb)**(-b)
        
        # Convert to cumulative distribution
        cumulative_rainfall = np.cumsum(intensities)
        cumulative_fractions = cumulative_rainfall / cumulative_rainfall[-1]
        
        return np.column_stack([time_fractions, cumulative_fractions])
    
    def generate_design_storm(self, total_rainfall: float,
                            duration_hours: float,
                            distribution_type: str = 'scs_type_ii') -> Dict[str, np.ndarray]:
        """Generate design storm hyetograph.
        
        Args:
            total_rainfall: Total rainfall depth in mm
            duration_hours: Storm duration in hours
            distribution_type: Type of rainfall distribution to use
            
        Returns:
            Dictionary containing:
            - time: Time array in hours
            - rainfall: Rainfall intensity array in mm/hr
            - cumulative: Cumulative rainfall array in mm
            
        Raises:
            ValueError: If invalid distribution type specified
        """
        if distribution_type not in self.storm_distributions:
            raise ValueError(f"Invalid distribution type: {distribution_type}")
        
        # Get distribution
        distribution = self.storm_distributions[distribution_type]
        time_fractions = distribution[:, 0]
        cumulative_fractions = distribution[:, 1]
        
        # Create time array
        num_steps = int(duration_hours / self.time_step_hours) + 1
        time_hours = np.linspace(0, duration_hours, num_steps)
        time_fractions_interp = time_hours / duration_hours
        
        # Interpolate cumulative rainfall
        interp_func = interpolate.interp1d(time_fractions, cumulative_fractions,
                                          kind='linear', bounds_error=False, fill_value=(0, 1))
        cumulative_rainfall = interp_func(time_fractions_interp) * total_rainfall
        
        # Calculate incremental rainfall
        incremental_rainfall = np.diff(cumulative_rainfall, prepend=0)
        
        # Convert to intensity (mm/hr)
        rainfall_intensity = incremental_rainfall / self.time_step_hours
        
        return {
            'time': time_hours,
            'rainfall': rainfall_intensity,
            'cumulative': cumulative_rainfall
        }
    
    def generate_noaa_atlas14_scenarios(self, precipitation_data: Dict[int, Dict[int, float]],
                                      return_periods: Optional[List[int]] = None,
                                      durations: Optional[List[int]] = None) -> Dict[str, Dict]:
        """Generate rainfall scenarios from NOAA Atlas 14 data.
        
        Args:
            precipitation_data: Nested dictionary {return_period: {duration: depth_mm}}
            return_periods: List of return periods to generate (default: all available)
            durations: List of durations to generate (default: all available)
            
        Returns:
            Dictionary of rainfall scenarios keyed by 'rp{return_period}_d{duration}h'
        """
        if return_periods is None:
            return_periods = list(precipitation_data.keys())
        
        scenarios = {}
        
        for rp in return_periods:
            if rp not in precipitation_data:
                warnings.warn(f"Return period {rp} not in precipitation data")
                continue
            
            rp_data = precipitation_data[rp]
            
            if durations is None:
                durations_to_use = list(rp_data.keys())
            else:
                durations_to_use = durations
            
            for duration in durations_to_use:
                if duration not in rp_data:
                    warnings.warn(f"Duration {duration}h not available for {rp}-year return period")
                    continue
                
                total_rainfall = rp_data[duration]
                
                # Generate scenario for each distribution type
                for dist_type in ['scs_type_ii', 'scs_type_i', 'uniform']:
                    scenario_key = f"rp{rp}_d{duration}h_{dist_type}"
                    
                    scenario = self.generate_design_storm(
                        total_rainfall=total_rainfall,
                        duration_hours=duration,
                        distribution_type=dist_type
                    )
                    
                    scenario['return_period'] = rp
                    scenario['duration'] = duration
                    scenario['distribution_type'] = dist_type
                    scenario['total_rainfall'] = total_rainfall
                    
                    scenarios[scenario_key] = scenario
        
        return scenarios
    
    def generate_synthetic_storm(self, mean_intensity: float,
                               duration_hours: float,
                               peak_factor: float = 2.0,
                               randomness: float = 0.1) -> Dict[str, np.ndarray]:
        """Generate synthetic storm with random variability.
        
        Args:
            mean_intensity: Mean rainfall intensity in mm/hr
            duration_hours: Storm duration in hours
            peak_factor: Factor by which peak intensity exceeds mean
            randomness: Degree of random variability (0-1)
            
        Returns:
            Dictionary containing time, rainfall, and cumulative arrays
        """
        # Create time array
        num_steps = int(duration_hours / self.time_step_hours) + 1
        time_hours = np.linspace(0, duration_hours, num_steps)
        
        # Generate base intensity pattern (triangular)
        peak_time = duration_hours * 0.4  # Peak at 40% of duration
        intensities = np.zeros(num_steps)
        
        for i, t in enumerate(time_hours):
            if t <= peak_time:
                intensities[i] = mean_intensity * peak_factor * (t / peak_time)
            else:
                intensities[i] = mean_intensity * peak_factor * (duration_hours - t) / (duration_hours - peak_time)
        
        # Add random variability
        if randomness > 0:
            noise = np.random.normal(1, randomness, num_steps)
            noise = np.maximum(noise, 0.1)  # Prevent negative values
            intensities *= noise
        
        # Normalize to maintain total rainfall
        total_rainfall = mean_intensity * duration_hours
        current_total = np.sum(intensities) * self.time_step_hours
        intensities *= total_rainfall / current_total
        
        # Calculate cumulative rainfall
        cumulative_rainfall = np.cumsum(intensities) * self.time_step_hours
        
        return {
            'time': time_hours,
            'rainfall': intensities,
            'cumulative': cumulative_rainfall
        }
    
    def generate_multiple_storm_ensemble(self, base_scenario: Dict[str, np.ndarray],
                                       num_realizations: int = 100,
                                       intensity_variability: float = 0.2,
                                       timing_variability: float = 0.1) -> List[Dict[str, np.ndarray]]:
        """Generate ensemble of storm realizations with variability.
        
        Args:
            base_scenario: Base storm scenario to perturb
            num_realizations: Number of ensemble members
            intensity_variability: Standard deviation of intensity perturbations
            timing_variability: Standard deviation of timing perturbations (fraction)
            
        Returns:
            List of storm scenarios with variability
        """
        ensemble = []
        base_rainfall = base_scenario['rainfall']
        base_time = base_scenario['time']
        
        for i in range(num_realizations):
            # Perturb intensities
            intensity_multipliers = np.random.lognormal(
                mean=0, sigma=intensity_variability, size=len(base_rainfall)
            )
            perturbed_rainfall = base_rainfall * intensity_multipliers
            
            # Perturb timing (shift storm)
            if timing_variability > 0:
                duration = base_time[-1]
                time_shift = np.random.normal(0, timing_variability * duration)
                time_shift = np.clip(time_shift, -duration/4, duration/4)  # Limit shift
                
                # Apply time shift
                shifted_time = base_time + time_shift
                
                # Interpolate back to original time grid
                valid_mask = (shifted_time >= 0) & (shifted_time <= duration)
                
                if np.sum(valid_mask) > 1:
                    interp_func = interpolate.interp1d(
                        shifted_time[valid_mask],
                        perturbed_rainfall[valid_mask],
                        kind='linear',
                        bounds_error=False,
                        fill_value=0
                    )
                    perturbed_rainfall = interp_func(base_time)
                
            # Ensure non-negative rainfall
            perturbed_rainfall = np.maximum(perturbed_rainfall, 0)
            
            # Normalize to preserve total rainfall
            if np.sum(perturbed_rainfall) > 0:
                original_total = np.sum(base_rainfall)
                current_total = np.sum(perturbed_rainfall)
                perturbed_rainfall *= original_total / current_total
            
            # Calculate cumulative rainfall
            cumulative_rainfall = np.cumsum(perturbed_rainfall) * self.time_step_hours
            
            ensemble.append({
                'time': base_time.copy(),
                'rainfall': perturbed_rainfall,
                'cumulative': cumulative_rainfall,
                'realization': i
            })
        
        return ensemble
    
    def save_scenarios(self, scenarios: Dict[str, Dict],
                      output_path: Union[str, Path]) -> None:
        """Save rainfall scenarios to files.
        
        Args:
            scenarios: Dictionary of rainfall scenarios
            output_path: Output directory path
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for scenario_name, scenario_data in scenarios.items():
            # Save as CSV
            df = pd.DataFrame({
                'time_hours': scenario_data['time'],
                'rainfall_mm_per_hr': scenario_data['rainfall'],
                'cumulative_mm': scenario_data['cumulative']
            })
            
            csv_file = output_path / f"{scenario_name}.csv"
            df.to_csv(csv_file, index=False)
            
            # Save metadata as JSON
            metadata = {
                'scenario_name': scenario_name,
                'return_period': scenario_data.get('return_period'),
                'duration': scenario_data.get('duration'),
                'distribution_type': scenario_data.get('distribution_type'),
                'total_rainfall': scenario_data.get('total_rainfall'),
                'time_step_minutes': self.time_step_minutes,
                'max_intensity': float(np.max(scenario_data['rainfall'])),
                'mean_intensity': float(np.mean(scenario_data['rainfall']))
            }
            
            json_file = output_path / f"{scenario_name}_metadata.json"
            with open(json_file, 'w') as f:
                json.dump(metadata, f, indent=2)
    
    def load_noaa_atlas14_data(self, data_file: Union[str, Path]) -> Dict[int, Dict[int, float]]:
        """Load NOAA Atlas 14 precipitation data from file.
        
        Args:
            data_file: Path to CSV file with columns: return_period, duration_hours, precipitation_mm
            
        Returns:
            Nested dictionary of precipitation data
        """
        df = pd.read_csv(data_file)
        
        # Validate required columns
        required_cols = ['return_period', 'duration_hours', 'precipitation_mm']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Convert to nested dictionary
        precipitation_data = {}
        
        for _, row in df.iterrows():
            rp = int(row['return_period'])
            duration = int(row['duration_hours'])
            precip = float(row['precipitation_mm'])
            
            if rp not in precipitation_data:
                precipitation_data[rp] = {}
            
            precipitation_data[rp][duration] = precip
        
        return precipitation_data
    
    def validate_scenarios(self, scenarios: Dict[str, Dict]) -> Dict[str, Dict[str, bool]]:
        """Validate generated rainfall scenarios.
        
        Args:
            scenarios: Dictionary of rainfall scenarios
            
        Returns:
            Dictionary of validation results for each scenario
        """
        validation_results = {}
        
        for scenario_name, scenario_data in scenarios.items():
            validation = {}
            
            time_data = scenario_data['time']
            rainfall_data = scenario_data['rainfall']
            cumulative_data = scenario_data['cumulative']
            
            # Check for monotonic time
            validation['time_monotonic'] = np.all(np.diff(time_data) >= 0)
            
            # Check for non-negative rainfall
            validation['rainfall_non_negative'] = np.all(rainfall_data >= 0)
            
            # Check cumulative is monotonic
            validation['cumulative_monotonic'] = np.all(np.diff(cumulative_data) >= 0)
            
            # Check mass balance
            expected_cumulative = np.cumsum(rainfall_data) * self.time_step_hours
            validation['mass_balance'] = np.allclose(
                cumulative_data[1:], expected_cumulative[1:], rtol=1e-6
            )
            
            validation_results[scenario_name] = validation
        
        return validation_results