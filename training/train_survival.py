
"""
Survival analysis training script for crash prediction
Cox Proportional Hazards model on 7.8M crash records
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import yaml
import logging
from pathlib import Path
from datetime import datetime
import json
import pickle

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.survival_analysis import VehicleSafetySurvivalAnalysis
from src.data.preprocessor import TabularDataPreprocessor, WeatherDataProcessor
from src.utils.logger import setup_logger
import mlflow
import mlflow.sklearn

def load_crash_data(data_path: str, sample_size: int = None) -> pd.DataFrame:
    """
    Load and preprocess crash data
    
    Args:
        data_path: Path to crash data CSV
        sample_size: Optional sample size for development
        
    Returns:
        Processed crash data
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading crash data from {data_path}")
    
    # Load data in chunks if it's large
    chunk_size = 100000
    chunks = []
    
    try:
        for chunk in pd.read_csv(data_path, chunksize=chunk_size):
            chunks.append(chunk)
            if sample_size and len(chunks) * chunk_size >= sample_size:
                break
        
        data = pd.concat(chunks, ignore_index=True)
        
        if sample_size and len(data) > sample_size:
            data = data.sample(n=sample_size, random_state=42)
        
        logger.info(f"Loaded {len(data)} crash records")
        return data
        
    except Exception as e:
        logger.error(f"Failed to load crash data: {e}")
        raise

def engineer_survival_features(crash_data: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features for survival analysis
    
    Args:
        crash_data: Raw crash data
        
    Returns:
        Data with engineered features
    """
    logger = logging.getLogger(__name__)
    logger.info("Engineering survival analysis features...")
    
    data = crash_data.copy()
    
    # Temporal features
    if 'crash_date' in data.columns:
        data['crash_date'] = pd.to_datetime(data['crash_date'])
        data['crash_hour'] = data['crash_date'].dt.hour
        data['crash_day_of_week'] = data['crash_date'].dt.dayofweek
        data['crash_month'] = data['crash_date'].dt.month
        data['crash_quarter'] = data['crash_date'].dt.quarter
        data['is_weekend'] = data['crash_day_of_week'].isin([5, 6]).astype(int)
        data['is_night'] = ((data['crash_hour'] >= 22) | (data['crash_hour'] <= 6)).astype(int)
        data['is_rush_hour'] = ((data['crash_hour'].between(7, 9)) | 
                               (data['crash_hour'].between(17, 19))).astype(int)
    
    # Vehicle features
    if 'vehicle_age' in data.columns:
        data['vehicle_age_squared'] = data['vehicle_age'] ** 2
        data['vehicle_age_bins'] = pd.cut(data['vehicle_age'], 
                                        bins=[0, 3, 7, 15, 50], 
                                        labels=['new', 'medium', 'old', 'very_old'])
    
    # Driver features
    if 'driver_age' in data.columns:
        data['driver_age_squared'] = data['driver_age'] ** 2
        data['is_young_driver'] = (data['driver_age'] < 25).astype(int)
        data['is_senior_driver'] = (data['driver_age'] > 65).astype(int)
    
    # Road condition features
    if 'road_surface' in data.columns:
        data['is_wet_road'] = data['road_surface'].isin(['wet', 'icy', 'snow']).astype(int)
    
    # Weather severity
    weather_processor = WeatherDataProcessor()
    if 'weather_condition' in data.columns:
        weather_data = data[['weather_condition']].copy()
        if 'visibility_km' in data.columns:
            weather_data['visibility_km'] = data['visibility_km']
        if 'wind_speed_kmh' in data.columns:
            weather_data['wind_speed_kmh'] = data['wind_speed_kmh']
        if 'temperature_c' in data.columns:
            weather_data['temperature_c'] = data['temperature_c']
            
        processed_weather = weather_processor.process_weather_conditions(weather_data)
        data['weather_severity'] = processed_weather['weather_severity']
    
    # Speed and violation features
    if 'speed_limit' in data.columns and 'estimated_speed' in data.columns:
        data['speed_over_limit'] = np.maximum(0, data['estimated_speed'] - data['speed_limit'])
        data['speed_ratio'] = data['estimated_speed'] / data['speed_limit']
    
    # Previous violations
    if 'previous_violations' in data.columns:
        data['has_violations'] = (data['previous_violations'] > 0).astype(int)
        data['violation_log'] = np.log1p(data['previous_violations'])
    
    # Location risk features
    if 'location_type' in data.columns:
        high_risk_locations = ['intersection', 'highway_merge', 'school_zone']
        data['is_high_risk_location'] = data['location_type'].isin(high_risk_locations).astype(int)
    
    # Interaction features
    if 'is_weekend' in data.columns and 'is_night' in data.columns:
        data['weekend_night'] = data['is_weekend'] * data['is_night']
    
    if 'weather_severity' in data.columns and 'is_night' in data.columns:
        data['weather_night_interaction'] = data['weather_severity'] * data['is_night']
    
    logger.info(f"Feature engineering completed. Shape: {data.shape}")
    return data

def prepare_survival_data(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Prepare data for survival analysis
    
    Args:
        data: Engineered crash data
        config: Configuration dictionary
        
    Returns:
        Survival analysis ready data
    """
    logger = logging.getLogger(__name__)
    logger.info("Preparing survival analysis data...")
    
    # Extract configuration
    duration_col = config.get('survival_duration_column', 'time_to_crash')
    event_col = config.get('survival_event_column', 'crash_occurred')
    min_time = config.get('min_survival_time', 0.1)
    max_time = config.get('max_survival_time', 60)
    
    # Filter data
    survival_data = data.copy()
    
    # Ensure positive survival times
    if duration_col in survival_data.columns:
        survival_data = survival_data[survival_data[duration_col] > min_time]
        survival_data = survival_data[survival_data[duration_col] <= max_time]
    
    # Ensure event column is binary
    if event_col in survival_data.columns:
        survival_data[event_col] = survival_data[event_col].astype(bool)
    
    # Remove rows with missing survival data
    survival_data = survival_data.dropna(subset=[duration_col, event_col])
    
    logger.info(f"Survival data prepared. Shape: {survival_data.shape}")
    logger.info(f"Event rate: {survival_data[event_col].mean():.3f}")
    
    return survival_data

def train_survival_model(survival_data: pd.DataFrame, config: dict, output_dir: str):
    """
    Train Cox Proportional Hazards model
    
    Args:
        survival_data: Prepared survival data
        config: Training configuration
        output_dir: Output directory
        
    Returns:
        Trained model and metrics
    """
    logger = logging.getLogger(__name__)
    logger.info("Training survival analysis model...")
    
    # Initialize model
    cox_config = config.get('cox_model', {})
    model = VehicleSafetySurvivalAnalysis(
        penalizer=cox_config.get('penalizer', 0.01),
        l1_ratio=cox_config.get('l1_ratio', 0.1),
        alpha_level=cox_config.get('alpha', 0.05)
    )
    
    # Prepare data
    duration_col = config.get('survival_duration_column', 'time_to_crash')
    event_col = config.get('survival_event_column', 'crash_occurred')
    
    # Feature columns (exclude target and metadata)
    exclude_cols = [duration_col, event_col, 'crash_id', 'vehicle_id', 'driver_id', 'crash_date']
    feature_cols = [col for col in survival_data.columns if col not in exclude_cols]
    
    prepared_data = model.prepare_survival_data(
        survival_data,
        duration_col=duration_col,
        event_col=event_col,
        feature_cols=feature_cols
    )
    
    # Train model
    validation_split = config.get('validation_split', 0.2)
    training_metrics = model.fit(prepared_data, validation_split=validation_split)
    
    # Get hazard ratios
    hazard_ratios = model.get_hazard_ratios()
    
    # Save model
    model_path = os.path.join(output_dir, 'survival_model.pkl')
    model.save_model(model_path)
    
    logger.info(f"Model saved to {model_path}")
    logger.info(f"Training metrics: {training_metrics}")
    
    return model, training_metrics, hazard_ratios, model_path

def evaluate_survival_model(model: VehicleSafetySurvivalAnalysis, 
                          test_data: pd.DataFrame,
                          config: dict) -> dict:
    """
    Evaluate survival model performance
    
    Args:
        model: Trained survival model
        test_data: Test dataset
        config: Evaluation configuration
        
    Returns:
        Evaluation metrics
    """
    logger = logging.getLogger(__name__)
    logger.info("Evaluating survival model...")
    
    # Prepare test data
    duration_col = config.get('survival_duration_column', 'time_to_crash')
    event_col = config.get('survival_event_column', 'crash_occurred')
    
    exclude_cols = [duration_col, event_col, 'crash_id', 'vehicle_id', 'driver_id', 'crash_date']
    feature_cols = [col for col in test_data.columns if col not in exclude_cols]
    
    test_prepared = model.prepare_survival_data(
        test_data,
        duration_col=duration_col,
        event_col=event_col,
        feature_cols=feature_cols
    )
    
    # Calculate metrics
    metrics = model._calculate_metrics(test_prepared, "Test")
    
    # Risk stratification analysis
    test_features = test_prepared[feature_cols]
    risk_strata = model.stratify_risk(test_features)
    
    # Risk distribution
    risk_distribution = risk_strata['risk_category'].value_counts(normalize=True).to_dict()
    
    # Time-dependent predictions
    time_points = config.get('time_horizon_months', [1, 3, 6, 12])
    crash_probs = model.predict_crash_probabilities(test_features, time_points)
    
    avg_crash_probs = crash_probs.mean().to_dict()
    
    evaluation_results = {
        'c_index': metrics['c_index'],
        'log_likelihood': metrics['log_likelihood'],
        'risk_distribution': risk_distribution,
        'average_crash_probabilities': avg_crash_probs,
        'test_samples': len(test_prepared)
    }
    
    logger.info(f"Evaluation results: {evaluation_results}")
    
    return evaluation_results

def estimate_business_impact(model: VehicleSafetySurvivalAnalysis,
                           test_data: pd.DataFrame,
                           config: dict) -> dict:
    """
    Estimate business impact of survival model
    
    Args:
        model: Trained survival model
        test_data: Test dataset
        config: Configuration
        
    Returns:
        Business impact estimates
    """
    logger = logging.getLogger(__name__)
    logger.info("Estimating business impact...")
    
    # Extract features
    duration_col = config.get('survival_duration_column', 'time_to_crash')
    event_col = config.get('survival_event_column', 'crash_occurred')
    exclude_cols = [duration_col, event_col, 'crash_id', 'vehicle_id', 'driver_id', 'crash_date']
    feature_cols = [col for col in test_data.columns if col not in exclude_cols]
    
    test_features = test_data[feature_cols]
    
    # Estimate crash prevention potential
    impact_estimates = model.estimate_crash_prevention(
        test_features,
        intervention_effect=0.2,  # 20% risk reduction from safety interventions
        time_horizon=12  # 12 months
    )
    
    # Scale to population
    population_scale = config.get('target_vehicle_coverage', 300000)
    sample_scale = len(test_features)
    scaling_factor = population_scale / sample_scale
    
    scaled_impact = {
        'crashes_prevented_annual': impact_estimates['crashes_prevented'] * scaling_factor,
        'economic_impact_millions': impact_estimates['economic_impact_usd'] * scaling_factor / 1e6,
        'prevention_rate': impact_estimates['prevention_rate'],
        'population_covered': population_scale
    }
    
    logger.info(f"Business impact estimates: {scaled_impact}")
    
    return scaled_impact

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train survival analysis model")
    parser.add_argument("--config", type=str, default="config/training_config.yaml",
                       help="Path to training configuration")
    parser.add_argument("--data", type=str, required=True,
                       help="Path to crash data CSV")
    parser.add_argument("--output", type=str, default="data/models/",
                       help="Output directory for trained model")
    parser.add_argument("--sample-size", type=int, default=None,
                       help="Sample size for development")
    parser.add_argument("--experiment-name", type=str, default="survival_analysis",
                       help="MLflow experiment name")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger("survival_training", level="INFO")
    logger.info("Starting survival analysis training")
    
    try:
        # Load configuration
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        survival_config = config.get('survival_training', {})
        data_config = config.get('data', {})
        eval_config = config.get('evaluation', {})
        
        # Setup MLflow
        mlflow.set_experiment(args.experiment_name)
        
        with mlflow.start_run():
            # Load data
            crash_data = load_crash_data(args.data, args.sample_size)
            
            # Log data info
            mlflow.log_params({
                'data_path': args.data,
                'sample_size': len(crash_data),
                'data_columns': len(crash_data.columns)
            })
            
            # Feature engineering
            engineered_data = engineer_survival_features(crash_data)
            
            # Prepare survival data
            survival_data = prepare_survival_data(engineered_data, survival_config)
            
            # Split data
            from sklearn.model_selection import train_test_split
            train_data, test_data = train_test_split(
                survival_data,
                test_size=0.2,
                random_state=42,
                stratify=survival_data[survival_config.get('survival_event_column', 'crash_occurred')]
            )
            
            logger.info(f"Train set: {len(train_data)}, Test set: {len(test_data)}")
            
            # Train model
            model, training_metrics, hazard_ratios, model_path = train_survival_model(
                train_data, survival_config, args.output
            )
            
            # Log training metrics
            mlflow.log_metrics(training_metrics)
            mlflow.log_artifact(model_path)
            
            # Evaluate model
            evaluation_results = evaluate_survival_model(model, test_data, survival_config)
            mlflow.log_metrics({f"test_{k}": v for k, v in evaluation_results.items() 
                               if isinstance(v, (int, float))})
            
            # Business impact
            business_impact = estimate_business_impact(model, test_data, config.get('business_impact', {}))
            mlflow.log_metrics({f"business_{k}": v for k, v in business_impact.items()
                               if isinstance(v, (int, float))})
            
            # Save comprehensive results
            results = {
                'model_path': model_path,
                'training_config': survival_config,
                'training_metrics': training_metrics,
                'evaluation_results': evaluation_results,
                'business_impact': business_impact,
                'hazard_ratios': hazard_ratios.to_dict('records'),
                'data_info': {
                    'total_samples': len(survival_data),
                    'train_samples': len(train_data),
                    'test_samples': len(test_data),
                    'event_rate': float(survival_data[survival_config.get('survival_event_column', 'crash_occurred')].mean())
                },
                'timestamp': datetime.now().isoformat()
            }
            
            results_file = os.path.join(args.output, 'survival_training_results.json')
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            mlflow.log_artifact(results_file)
            
            # Performance validation
            target_c_index = eval_config.get('target_metrics', {}).get('survival_c_index', 0.75)
            c_index_met = evaluation_results['c_index'] >= target_c_index
            
            logger.info(f"C-index target ({target_c_index}): {'✅ Met' if c_index_met else '❌ Not met'}")
            logger.info(f"Final C-index: {evaluation_results['c_index']:.4f}")
            
            # Log final status
            mlflow.log_params({
                'c_index_target_met': c_index_met,
                'final_c_index': evaluation_results['c_index']
            })
            
            logger.info("Survival analysis training completed successfully")
            
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
