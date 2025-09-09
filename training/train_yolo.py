
"""
YOLOv7 training script for vehicle detection
Custom training pipeline for Bridgestone vehicle safety dataset
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
from pathlib import Path
import logging
from datetime import datetime
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from ultralytics import YOLO
from src.utils.logger import setup_logger
from src.data.data_loader import VehicleDataLoader
from src.data.augmentation import VehicleImageAugmentation

def load_dataset_config(config_path: str) -> dict:
    """Load dataset configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def prepare_yolo_dataset(data_dir: str, output_dir: str, train_ratio: float = 0.8):
    """
    Prepare dataset in YOLO format
    
    Args:
        data_dir: Directory containing images and annotations
        output_dir: Output directory for YOLO dataset
        train_ratio: Ratio of training data
    """
    logger = logging.getLogger(__name__)
    
    # Create YOLO dataset structure
    dataset_path = Path(output_dir)
    (dataset_path / "images" / "train").mkdir(parents=True, exist_ok=True)
    (dataset_path / "images" / "val").mkdir(parents=True, exist_ok=True)
    (dataset_path / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (dataset_path / "labels" / "val").mkdir(parents=True, exist_ok=True)
    
    # Load and split data
    data_loader = VehicleDataLoader()
    
    # Process dataset
    image_files = list(Path(data_dir).glob("*.jpg")) + list(Path(data_dir).glob("*.png"))
    total_images = len(image_files)
    train_count = int(total_images * train_ratio)
    
    logger.info(f"Processing {total_images} images: {train_count} train, {total_images - train_count} val")
    
    for i, img_file in enumerate(image_files):
        is_train = i < train_count
        split = "train" if is_train else "val"
        
        # Copy image
        dst_img = dataset_path / "images" / split / img_file.name
        dst_img.write_bytes(img_file.read_bytes())
        
        # Convert annotation if exists
        ann_file = img_file.with_suffix('.txt')
        if ann_file.exists():
            dst_ann = dataset_path / "labels" / split / ann_file.name
            dst_ann.write_text(ann_file.read_text())
    
    # Create dataset.yaml
    dataset_yaml = {
        'path': str(dataset_path),
        'train': 'images/train',
        'val': 'images/val',
        'names': {
            0: 'person',
            1: 'bicycle', 
            2: 'car',
            3: 'motorcycle',
            4: 'airplane',
            5: 'bus',
            6: 'train',
            7: 'truck'
        }
    }
    
    with open(dataset_path / "dataset.yaml", 'w') as f:
        yaml.dump(dataset_yaml, f)
    
    logger.info(f"YOLO dataset prepared at {dataset_path}")
    return str(dataset_path / "dataset.yaml")

def train_yolo_model(config: dict, dataset_yaml: str, output_dir: str):
    """
    Train YOLOv7 model
    
    Args:
        config: Training configuration
        dataset_yaml: Path to dataset configuration
        output_dir: Output directory for trained model
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting YOLOv7 training...")
    
    # Initialize model
    model_size = config.get('model_size', 'yolov7')
    model = YOLO(f'{model_size}.pt')  # Load pretrained model
    
    # Training parameters
    training_params = {
        'data': dataset_yaml,
        'epochs': config.get('epochs', 300),
        'imgsz': config.get('input_size', [640, 640])[0],
        'batch': config.get('batch_size', 16),
        'lr0': config.get('learning_rate', 0.01),
        'momentum': config.get('momentum', 0.937),
        'weight_decay': config.get('weight_decay', 0.0005),
        'workers': config.get('workers', 8),
        'device': config.get('device', 'auto'),
        'project': output_dir,
        'name': 'vehicle_detection',
        'save': True,
        'save_period': config.get('checkpoint_frequency', 10),
        'val': True,
        'plots': True,
        'verbose': True
    }
    
    # Data augmentation parameters
    augmentation_params = {
        'mosaic': config.get('mosaic_prob', 1.0),
        'mixup': config.get('mixup_prob', 0.15),
        'copy_paste': config.get('copy_paste_prob', 0.3),
        'degrees': 0.0,  # rotation
        'translate': 0.1,  # translation
        'scale': 0.5,  # scaling
        'shear': 0.0,  # shearing
        'perspective': 0.0,  # perspective
        'flipud': 0.0,  # vertical flip
        'fliplr': 0.5,  # horizontal flip
        'hsv_h': 0.015,  # hue
        'hsv_s': 0.7,  # saturation
        'hsv_v': 0.4   # value
    }
    
    training_params.update(augmentation_params)
    
    logger.info(f"Training parameters: {training_params}")
    
    # Start training
    try:
        results = model.train(**training_params)
        
        # Save best model
        best_model_path = Path(output_dir) / 'vehicle_detection' / 'weights' / 'best.pt'
        final_model_path = Path(output_dir) / 'yolov7_vehicle_best.pt'
        
        if best_model_path.exists():
            # Copy best model to final location
            import shutil
            shutil.copy2(best_model_path, final_model_path)
            logger.info(f"Best model saved to {final_model_path}")
        
        return results, str(final_model_path)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

def evaluate_model(model_path: str, dataset_yaml: str, config: dict):
    """
    Evaluate trained model
    
    Args:
        model_path: Path to trained model
        dataset_yaml: Path to dataset configuration  
        config: Evaluation configuration
    """
    logger = logging.getLogger(__name__)
    logger.info("Evaluating trained model...")
    
    # Load trained model
    model = YOLO(model_path)
    
    # Run validation
    val_results = model.val(
        data=dataset_yaml,
        imgsz=config.get('input_size', [640, 640])[0],
        batch=config.get('batch_size', 16),
        conf=0.001,  # confidence threshold
        iou=0.6,     # IoU threshold for NMS
        device=config.get('device', 'auto'),
        plots=True,
        save_json=True
    )
    
    # Extract metrics
    metrics = {
        'mAP_0.5': float(val_results.box.map50),
        'mAP_0.5:0.95': float(val_results.box.map),
        'precision': float(val_results.box.mp),
        'recall': float(val_results.box.mr),
        'f1_score': 2 * (val_results.box.mp * val_results.box.mr) / (val_results.box.mp + val_results.box.mr),
    }
    
    logger.info(f"Evaluation results: {metrics}")
    
    # Performance validation
    target_map50 = config.get('target_map50', 0.6)
    target_precision = config.get('target_precision', 0.8)
    target_recall = config.get('target_recall', 0.75)
    
    performance_check = {
        'map50_met': metrics['mAP_0.5'] >= target_map50,
        'precision_met': metrics['precision'] >= target_precision,
        'recall_met': metrics['recall'] >= target_recall
    }
    
    logger.info(f"Performance targets: {performance_check}")
    
    return metrics, performance_check

def benchmark_inference(model_path: str, config: dict):
    """
    Benchmark inference performance
    
    Args:
        model_path: Path to trained model
        config: Benchmark configuration
    """
    logger = logging.getLogger(__name__)
    logger.info("Benchmarking inference performance...")
    
    import time
    import numpy as np
    
    # Load model
    model = YOLO(model_path)
    
    # Create dummy input
    input_size = config.get('input_size', [640, 640])
    dummy_image = np.random.randint(0, 255, (*input_size, 3), dtype=np.uint8)
    
    # Warmup
    for _ in range(10):
        _ = model.predict(dummy_image, verbose=False)
    
    # Benchmark
    num_iterations = 100
    inference_times = []
    
    for _ in range(num_iterations):
        start_time = time.time()
        _ = model.predict(dummy_image, verbose=False)
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        inference_times.append(inference_time)
    
    # Calculate statistics
    benchmark_results = {
        'mean_inference_time_ms': float(np.mean(inference_times)),
        'median_inference_time_ms': float(np.median(inference_times)),
        'p95_inference_time_ms': float(np.percentile(inference_times, 95)),
        'p99_inference_time_ms': float(np.percentile(inference_times, 99)),
        'min_inference_time_ms': float(np.min(inference_times)),
        'max_inference_time_ms': float(np.max(inference_times)),
        'std_inference_time_ms': float(np.std(inference_times)),
        'throughput_fps': 1000 / np.mean(inference_times)
    }
    
    logger.info(f"Benchmark results: {benchmark_results}")
    
    # Check SLA
    sla_target_ms = 150
    sla_compliance = benchmark_results['p95_inference_time_ms'] <= sla_target_ms
    
    logger.info(f"SLA compliance (P95 <= {sla_target_ms}ms): {sla_compliance}")
    
    return benchmark_results, sla_compliance

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train YOLOv7 for vehicle detection")
    parser.add_argument("--config", type=str, default="config/training_config.yaml",
                       help="Path to training configuration")
    parser.add_argument("--data-dir", type=str, required=True,
                       help="Path to dataset directory")
    parser.add_argument("--output-dir", type=str, default="runs/train",
                       help="Output directory for training results")
    parser.add_argument("--dataset-yaml", type=str, default=None,
                       help="Path to existing dataset.yaml file")
    parser.add_argument("--skip-training", action="store_true",
                       help="Skip training and only evaluate existing model")
    parser.add_argument("--model-path", type=str, default=None,
                       help="Path to existing model for evaluation")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger("yolo_training", level="INFO")
    logger.info("Starting YOLOv7 training pipeline")
    
    try:
        # Load configuration
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        yolo_config = config.get('yolo_training', {})
        eval_config = config.get('evaluation', {})
        
        # Prepare dataset if needed
        if args.dataset_yaml is None:
            logger.info("Preparing YOLO dataset...")
            dataset_yaml = prepare_yolo_dataset(
                args.data_dir, 
                os.path.join(args.output_dir, 'dataset'),
                train_ratio=0.8
            )
        else:
            dataset_yaml = args.dataset_yaml
        
        model_path = args.model_path
        
        # Training
        if not args.skip_training:
            logger.info("Starting model training...")
            training_results, model_path = train_yolo_model(
                yolo_config, 
                dataset_yaml,
                args.output_dir
            )
            
            logger.info("Training completed successfully")
        
        # Evaluation
        if model_path and os.path.exists(model_path):
            logger.info("Evaluating model...")
            metrics, performance_check = evaluate_model(
                model_path, 
                dataset_yaml, 
                eval_config
            )
            
            # Benchmark
            logger.info("Benchmarking inference...")
            benchmark_results, sla_compliance = benchmark_inference(
                model_path,
                yolo_config
            )
            
            # Save results
            results = {
                'model_path': model_path,
                'training_config': yolo_config,
                'evaluation_metrics': metrics,
                'performance_check': performance_check,
                'benchmark_results': benchmark_results,
                'sla_compliance': sla_compliance,
                'timestamp': datetime.now().isoformat()
            }
            
            results_file = os.path.join(args.output_dir, 'yolo_training_results.json')
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Results saved to {results_file}")
            
            # Final status
            if all(performance_check.values()) and sla_compliance:
                logger.info("✅ All performance targets met!")
            else:
                logger.warning("⚠️ Some performance targets not met")
                
        else:
            logger.error("No model found for evaluation")
            
        logger.info("YOLOv7 training pipeline completed")
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
