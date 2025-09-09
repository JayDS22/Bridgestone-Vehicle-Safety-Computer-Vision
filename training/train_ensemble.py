"""
Training script for Ensemble Risk Assessment Model
Orchestrates training of Random Forest, XGBoost, and Neural Network components
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import yaml
import json
from datetime import datetime
import mlflow
import mlflow.sklearn
import mlflow.pytorch

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.ensemble_model import EnsembleRiskModel
from src.models.feature_extractor import AdvancedFeatureExtractor
from src.utils.logger import setup_logger
from src.utils.metrics import PerformanceMonitor

def load_config(config_path: str) -> dict:
    """Load training configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_training_data(data_path: str, sample_size: int = None) -> pd.DataFrame:
    """
    Load and preprocess training data
    
    Args:
        data_path: Path to training data CSV
        sample_size: Optional sample size for development
        
    Returns:
        Processed DataFrame
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading training data from {data_path}")
    
    # Load data
    data = pd.read_csv(data_path)
    logger.info(f"Loaded {len(data)} samples with {len(data.columns)} features")
    
    # Sample data if specified
    if sample_size and sample_size < len(data):
        data = data.sample(n=sample_size, random_state=42)
        logger.info(f"Sampled {len(data)} samples for training")
    
    # Basic data validation
    logger.info(f"Missing values: {data.isnull().sum().sum()}")
    logger.info(f"Target distribution:\n{data['risk_label'].value_counts()}")
    
    return data

def create_features_and_labels(data: pd.DataFrame, target_col: str = 'risk_label') -> tuple:
    """
    Separate features and labels
    
    Args:
        data: Input DataFrame
        target_col: Name of target column
        
    Returns:
        Tuple of (features, labels)
    """
    # Identify feature columns (all except target and metadata)
    exclude_cols = [target_col, 'video_id', 'frame_id', 'timestamp', 'metadata']
    feature_cols = [col for col in data.columns if col not in exclude_cols]
    
    X = data[feature_cols]
    y = data[target_col]
    
    return X, y

def setup_mlflow(experiment_name: str = "bridgestone-vehicle-safety"):
    """Setup MLflow for experiment tracking"""
    mlflow.set_experiment(experiment_name)
    return mlflow.start_run()

def log_model_artifacts(model: EnsembleRiskModel, model_path: str):
    """Log model artifacts to MLflow"""
    # Save model locally first
    model.save_model(model_path)
    
    # Log to MLflow
    mlflow.log_artifact(model_path, "models")
    
    # Log model info
    model_info = model.get_model_info()
    mlflow.log_params(model_info)

def evaluate_model(model: EnsembleRiskModel, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Comprehensive model evaluation"""
    logger = logging.getLogger(__name__)
    logger.info("Evaluating model performance...")
    
    # Basic evaluation
    metrics = model.evaluate(X_test, y_test)
    
    # Feature importance
    importance = model.get_feature_importance()
    top_features = sorted(
        importance['ensemble'].items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:20]
    
    # Risk stratification analysis
    risk_scores = model.predict_risk_scores(X_test)
    risk_percentiles = np.percentile(risk_scores, [25, 50, 75, 90, 95, 99])
    
    evaluation_results = {
        **metrics,
        'top_features': dict(top_features),
        'risk_score_percentiles': {
            'p25': float(risk_percentiles[0]),
            'p50': float(risk_percentiles[1]),
            'p75': float(risk_percentiles[2]),
            'p90': float(risk_percentiles[3]),
            'p95': float(risk_percentiles[4]),
            'p99': float(risk_percentiles[5])
        }
    }
    
    return evaluation_results

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train Bridgestone Vehicle Safety Ensemble Model")
    parser.add_argument("--config", type=str, default="config/training_config.yaml",
                       help="Path to training configuration file")
    parser.add_argument("--data", type=str, required=True,
                       help="Path to training data CSV file")
    parser.add_argument("--output", type=str, default="data/models/ensemble_model.pkl",
                       help="Path to save trained model")
    parser.add_argument("--experiment", type=str, default="bridgestone-vehicle-safety",
                       help="MLflow experiment name")
    parser.add_argument("--sample-size", type=int, default=None,
                       help="Sample size for development (optional)")
    parser.add_argument("--validation-split", type=float, default=0.2,
                       help="Validation split ratio")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger("training", level=args.log_level)
    logger.info("Starting Ensemble Model Training")
    logger.info(f"Arguments: {vars(args)}")
    
    try:
        # Load configuration
        if os.path.exists(args.config):
            config = load_config(args.config)
            logger.info(f"Loaded configuration from {args.config}")
        else:
            config = {}
            logger.warning(f"Configuration file {args.config} not found, using defaults")
        
        # Setup MLflow
        mlflow_run = setup_mlflow(args.experiment)
        logger.info(f"Started MLflow run: {mlflow_run.info.run_id}")
        
        # Load training data
        data = load_training_data(args.data, args.sample_size)
        
        # Create features and labels
        X, y = create_features_and_labels(data)
        logger.info(f"Created feature matrix: {X.shape}")
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=args.validation_split,
            random_state=42,
            stratify=y
        )
        
        logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
        
        # Initialize model
        model_config = config.get('ensemble_model', {})
        model = EnsembleRiskModel(**model_config)
        
        # Log training parameters
        training_params = {
            'data_path': args.data,
            'sample_size': args.sample_size or len(data),
            'validation_split': args.validation_split,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'feature_count': X.shape[1],
            **model_config
        }
        mlflow.log_params(training_params)
        
        # Train model
        logger.info("Starting model training...")
        training_start = datetime.now()
        
        training_metrics = model.train(
            X_train, y_train,
            validation_split=0.2,  # Internal validation for neural network
            epochs=config.get('training', {}).get('epochs', 100),
            batch_size=config.get('training', {}).get('batch_size', 256)
        )
        
        training_duration = (datetime.now() - training_start).total_seconds()
        logger.info(f"Training completed in {training_duration:.2f} seconds")
        
        # Log training metrics
        mlflow.log_metrics(training_metrics)
        mlflow.log_metric('training_duration_seconds', training_duration)
        
        # Evaluate model
        evaluation_results = evaluate_model(model, X_test, y_test)
        logger.info(f"Test AUC: {evaluation_results['auc']:.4f}")
        logger.info(f"Test Accuracy: {evaluation_results['accuracy']:.4f}")
        
        # Log evaluation metrics
        mlflow.log_metrics({f"test_{k}": v for k, v in evaluation_results.items() 
                           if isinstance(v, (int, float))})
        
        # Save model
        model_dir = Path(args.output).parent
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model.save_model(args.output)
        logger.info(f"Model saved to {args.output}")
        
        # Log model artifacts
        log_model_artifacts(model, args.output)
        
        # Save training report
        report = {
            'training_info': {
                'timestamp': datetime.now().isoformat(),
                'duration_seconds': training_duration,
                'data_path': args.data,
                'model_path': args.output,
                'mlflow_run_id': mlflow_run.info.run_id
            },
            'data_info': {
                'total_samples': len(data),
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'feature_count': X.shape[1],
                'class_distribution': y.value_counts().to_dict()
            },
            'training_metrics': training_metrics,
            'evaluation_results': evaluation_results,
            'model_config': model_config
        }
        
        report_path = args.output.replace('.pkl', '_training_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        mlflow.log_artifact(report_path, "reports")
        logger.info(f"Training report saved to {report_path}")
        
        # Performance validation
        if evaluation_results['auc'] >= 0.85:
            logger.info("✅ Model meets performance requirements (AUC >= 0.85)")
        else:
            logger.warning(f"⚠️ Model AUC {evaluation_results['auc']:.4f} below target 0.85")
        
        if evaluation_results['accuracy'] >= 0.85:
            logger.info("✅ Model meets accuracy requirements (>= 85%)")
        else:
            logger.warning(f"⚠️ Model accuracy {evaluation_results['accuracy']:.4f} below target 0.85")
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        mlflow.log_param("training_status", "failed")
        mlflow.log_param("error_message", str(e))
        raise
    
    finally:
        mlflow.end_run()

if __name__ == "__main__":
    main()
