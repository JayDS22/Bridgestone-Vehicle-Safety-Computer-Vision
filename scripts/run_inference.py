#!/usr/bin/env python3
"""
Batch inference script for Bridgestone Vehicle Safety System
Process multiple images/videos for safety analysis
"""

import os
import sys
import argparse
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import json
import time
from datetime import datetime
from typing import List, Dict, Optional
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.yolo_detector import VehicleDetector
from src.models.feature_extractor import AdvancedFeatureExtractor
from src.models.ensemble_model import EnsembleRiskModel
from src.models.survival_analysis import VehicleSafetySurvivalAnalysis
from src.utils.logger import setup_logger
from src.utils.visualization import VehicleDetectionVisualizer
from src.data.data_loader import VehicleDataLoader

class BatchInferenceProcessor:
    """
    Batch processing system for vehicle safety inference
    """
    
    def __init__(self, 
                 model_config: Dict,
                 output_dir: str = "results/",
                 save_visualizations: bool = True):
        """
        Initialize batch processor
        
        Args:
            model_config: Model configuration dictionary
            output_dir: Output directory for results
            save_visualizations: Whether to save annotated images
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_visualizations = save_visualizations
        
        # Initialize models
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing models...")
        
        try:
            # Load detection model
            self.detector = VehicleDetector(
                model_path=model_config.get('yolo', {}).get('model_path', 'yolov7.pt'),
                confidence_threshold=model_config.get('yolo', {}).get('confidence_threshold', 0.5),
                device=model_config.get('yolo', {}).get('device', 'auto')
            )
            
            # Load feature extractor
            self.feature_extractor = AdvancedFeatureExtractor(
                device=model_config.get('feature_extraction', {}).get('device', 'auto')
            )
            
            # Load ensemble model
            self.ensemble_model = EnsembleRiskModel()
            ensemble_path = model_config.get('ensemble_model', {}).get('model_path', 'data/models/ensemble_model.pkl')
            if os.path.exists(ensemble_path):
                self.ensemble_model.load_model(ensemble_path)
            else:
                self.logger.warning(f"Ensemble model not found at {ensemble_path}")
                self.ensemble_model = None
            
            # Load survival model
            self.survival_model = VehicleSafetySurvivalAnalysis()
            survival_path = model_config.get('survival_analysis', {}).get('model_path', 'data/models/survival_model.pkl')
            if os.path.exists(survival_path):
                self.survival_model.load_model(survival_path)
            else:
                self.logger.warning(f"Survival model not found at {survival_path}")
                self.survival_model = None
            
            # Initialize visualizer
            if save_visualizations:
                self.visualizer = VehicleDetectionVisualizer()
            
            self.logger.info("Models initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize models: {e}")
            raise
    
    def process_single_image(self, 
                           image_path: str,
                           metadata: Optional[Dict] = None) -> Dict:
        """
        Process a single image for vehicle safety analysis
        
        Args:
            image_path: Path to image file
            metadata: Optional metadata dictionary
            
        Returns:
            Analysis results dictionary
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Step 1: Vehicle Detection
            start_time = time.time()
            detection_result = self.detector.detect(image_rgb, return_crops=False)
            detection_time = (time.time() - start_time) * 1000
            
            # Step 2: Feature Extraction
            start_time = time.time()
            features = self.feature_extractor.extract_features(
                image_rgb, 
                detection_result['detections'],
                metadata
            )
            feature_time = (time.time() - start_time) * 1000
            
            # Step 3: Risk Assessment
            risk_score = 0.5  # Default
            risk_time = 0
            if self.ensemble_model and self.ensemble_model.is_trained:
                start_time = time.time()
                feature_array = np.array(list(features.values())).reshape(1, -1)
                risk_score = self.ensemble_model.predict_risk_scores(feature_array)[0]
                risk_time = (time.time() - start_time) * 1000
            
            # Step 4: Survival Analysis
            crash_probabilities = {}
            survival_time = 0
            if self.survival_model and self.survival_model.is_fitted:
                start_time = time.time()
                feature_array = np.array(list(features.values())).reshape(1, -1)
                crash_probs = self.survival_model.predict_crash_probabilities(
                    feature_array, time_points=[1, 3, 6, 12]
                )
                crash_probabilities = crash_probs.iloc[0].to_dict()
                survival_time = (time.time() - start_time) * 1000
            
            # Compile results
            result = {
                'image_path': image_path,
                'timestamp': datetime.now().isoformat(),
                'detections': detection_result['detections'],
                'detection_count': len(detection_result['detections']['boxes']),
                'risk_score': float(risk_score),
                'crash_probabilities': {
                    key.split('_')[-1]: float(value) 
                    for key, value in crash_probabilities.items()
                } if crash_probabilities else {},
                'processing_times': {
                    'detection_ms': detection_time,
                    'feature_extraction_ms': feature_time,
                    'risk_assessment_ms': risk_time,
                    'survival_analysis_ms': survival_time,
                    'total_ms': detection_time + feature_time + risk_time + survival_time
                },
                'metadata': metadata or {}
            }
            
            # Save visualization
            if self.save_visualizations and hasattr(self, 'visualizer'):
                output_image_path = self.output_dir / "visualizations" / f"{Path(image_path).stem}_annotated.jpg"
                output_image_path.parent.mkdir(parents=True, exist_ok=True)
                
                annotated_image = self.visualizer.visualize_detections(
                    image_rgb,
                    detection_result['detections'],
                    risk_score=risk_score,
                    save_path=str(output_image_path)
                )
                
                result['visualization_path'] = str(output_image_path)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to process image {image_path}: {e}")
            return {
                'image_path': image_path,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def process_image_batch(self, 
                          image_paths: List[str],
                          batch_metadata: Optional[List[Dict]] = None,
                          max_workers: int = 4) -> List[Dict]:
        """
        Process batch of images
        
        Args:
            image_paths: List of image file paths
            batch_metadata: Optional list of metadata dictionaries
            max_workers: Maximum number of worker threads
            
        Returns:
            List of analysis results
        """
        self.logger.info(f"Processing batch of {len(image_paths)} images")
        
        results = []
        
        # Process images with progress bar
        for i, image_path in enumerate(tqdm(image_paths, desc="Processing images")):
            metadata = batch_metadata[i] if batch_metadata and i < len(batch_metadata) else None
            result = self.process_single_image(image_path, metadata)
            results.append(result)
        
        # Save batch results
        batch_results_path = self.output_dir / f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(batch_results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Batch results saved to {batch_results_path}")
        
        return results
    
    def process_video(self, 
                     video_path: str,
                     frame_skip: int = 1,
                     max_frames: Optional[int] = None) -> Dict:
        """
        Process video file for safety analysis
        
        Args:
            video_path: Path to video file
            frame_skip: Process every nth frame
            max_frames: Maximum number of frames to process
            
        Returns:
            Video analysis results
        """
        self.logger.info(f"Processing video: {video_path}")
        
        try:
            # Process video frames
            video_results = self.detector.detect_video(
                video_path=video_path,
                frame_skip=frame_skip
            )
            
            if max_frames:
                video_results = video_results[:max_frames]
            
            # Analyze each frame
            frame_analyses = []
            risk_scores = []
            
            for i, frame_result in enumerate(tqdm(video_results, desc="Analyzing frames")):
                # Extract features for this frame
                # Note: We don't have the actual frame image here, so we'll use detection-based features
                detection_features = {
                    'detection_count': len(frame_result['detections']['boxes']),
                    'avg_confidence': np.mean(frame_result['detections']['scores']) if frame_result['detections']['scores'] else 0,
                    'max_confidence': np.max(frame_result['detections']['scores']) if frame_result['detections']['scores'] else 0,
                    'vehicle_types': len(set(frame_result['detections']['labels'])),
                    'high_confidence_detections': sum(1 for score in frame_result['detections']['scores'] if score > 0.8)
                }
                
                # Simple risk assessment based on detection features
                risk_score = min(1.0, (
                    detection_features['detection_count'] * 0.1 +
                    (1 - detection_features['avg_confidence']) * 0.3 +
                    detection_features['vehicle_types'] * 0.2
                ))
                
                risk_scores.append(risk_score)
                
                frame_analysis = {
                    'frame_number': frame_result['frame_number'],
                    'timestamp': frame_result['timestamp'],
                    'detections': frame_result['detections'],
                    'risk_score': risk_score,
                    'detection_features': detection_features
                }
                
                frame_analyses.append(frame_analysis)
            
            # Video-level analysis
            video_analysis = {
                'video_path': video_path,
                'total_frames_processed': len(frame_analyses),
                'average_risk_score': float(np.mean(risk_scores)) if risk_scores else 0,
                'max_risk_score': float(np.max(risk_scores)) if risk_scores else 0,
                'high_risk_frames': sum(1 for score in risk_scores if score > 0.7),
                'risk_score_trend': risk_scores,
                'frame_analyses': frame_analyses,
                'processing_summary': {
                    'total_detections': sum(len(frame['detections']['boxes']) for frame in frame_analyses),
                    'unique_labels': list(set([
                        label for frame in frame_analyses 
                        for label in frame['detections']['labels']
                    ])),
                    'avg_processing_time_ms': np.mean([
                        frame.get('inference_time_ms', 0) for frame in video_results
                    ])
                },
                'timestamp': datetime.now().isoformat()
            }
            
            # Save video results
            video_results_path = self.output_dir / f"video_analysis_{Path(video_path).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(video_results_path, 'w') as f:
                json.dump(video_analysis, f, indent=2, default=str)
            
            self.logger.info(f"Video analysis saved to {video_results_path}")
            
            return video_analysis
            
        except Exception as e:
            self.logger.error(f"Failed to process video {video_path}: {e}")
            return {
                'video_path': video_path,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def generate_summary_report(self, results: List[Dict]) -> Dict:
        """
        Generate summary report from batch results
        
        Args:
            results: List of processing results
            
        Returns:
            Summary report dictionary
        """
        successful_results = [r for r in results if 'error' not in r]
        
        if not successful_results:
            return {'error': 'No successful results to summarize'}
        
        # Calculate statistics
        risk_scores = [r['risk_score'] for r in successful_results]
        detection_counts = [r['detection_count'] for r in successful_results]
        processing_times = [r['processing_times']['total_ms'] for r in successful_results]
        
        # Risk distribution
        risk_distribution = {
            'low_risk': sum(1 for score in risk_scores if score < 0.3),
            'medium_risk': sum(1 for score in risk_scores if 0.3 <= score < 0.7),
            'high_risk': sum(1 for score in risk_scores if score >= 0.7)
        }
        
        # Performance metrics
        sla_compliance = sum(1 for time_ms in processing_times if time_ms < 150) / len(processing_times)
        
        summary = {
            'total_processed': len(results),
            'successful': len(successful_results),
            'failed': len(results) - len(successful_results),
            'risk_statistics': {
                'mean_risk_score': float(np.mean(risk_scores)),
                'median_risk_score': float(np.median(risk_scores)),
                'std_risk_score': float(np.std(risk_scores)),
                'risk_distribution': risk_distribution
            },
            'detection_statistics': {
                'mean_detections': float(np.mean(detection_counts)),
                'max_detections': int(np.max(detection_counts)),
                'total_detections': int(np.sum(detection_counts))
            },
            'performance_statistics': {
                'mean_processing_time_ms': float(np.mean(processing_times)),
                'median_processing_time_ms': float(np.median(processing_times)),
                'p95_processing_time_ms': float(np.percentile(processing_times, 95)),
                'sla_compliance_rate': float(sla_compliance),
                'throughput_images_per_second': 1000 / np.mean(processing_times)
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return summary


def main():
    """Main function for batch inference"""
    parser = argparse.ArgumentParser(description="Batch inference for vehicle safety analysis")
    parser.add_argument("--input", type=str, required=True,
                       help="Input directory or file path")
    parser.add_argument("--output", type=str, default="results/",
                       help="Output directory")
    parser.add_argument("--config", type=str, default="config/model_config.yaml",
                       help="Model configuration file")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for processing")
    parser.add_argument("--visualizations", action="store_true",
                       help="Save visualization images")
    parser.add_argument("--video", action="store_true",
                       help="Process as video file")
    parser.add_argument("--frame-skip", type=int, default=1,
                       help="Frame skip for video processing")
    parser.add_argument("--max-frames", type=int, default=None,
                       help="Maximum frames to process from video")
    parser.add_argument("--recursive", action="store_true",
                       help="Process files recursively")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger("batch_inference", level="INFO")
    logger.info("Starting batch inference processing")
    
    try:
        # Load configuration
        import yaml
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Initialize processor
        processor = BatchInferenceProcessor(
            model_config=config,
            output_dir=args.output,
            save_visualizations=args.visualizations
        )
        
        input_path = Path(args.input)
        
        if args.video:
            # Process single video
            if input_path.is_file():
                result = processor.process_video(
                    str(input_path),
                    frame_skip=args.frame_skip,
                    max_frames=args.max_frames
                )
                logger.info(f"Video processing completed: {result.get('total_frames_processed', 0)} frames")
            else:
                logger.error("Video mode requires a single video file")
                return
        
        else:
            # Process images
            if input_path.is_file():
                # Single image
                image_paths = [str(input_path)]
            elif input_path.is_dir():
                # Directory of images
                image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
                if args.recursive:
                    image_paths = [
                        str(p) for p in input_path.rglob('*') 
                        if p.suffix.lower() in image_extensions
                    ]
                else:
                    image_paths = [
                        str(p) for p in input_path.glob('*') 
                        if p.suffix.lower() in image_extensions
                    ]
            else:
                logger.error(f"Input path not found: {input_path}")
                return
            
            if not image_paths:
                logger.error("No image files found")
                return
            
            logger.info(f"Found {len(image_paths)} images to process")
            
            # Process in batches
            all_results = []
            for i in range(0, len(image_paths), args.batch_size):
                batch_paths = image_paths[i:i + args.batch_size]
                logger.info(f"Processing batch {i//args.batch_size + 1}/{(len(image_paths) + args.batch_size - 1)//args.batch_size}")
                
                batch_results = processor.process_image_batch(batch_paths)
                all_results.extend(batch_results)
            
            # Generate summary report
            summary = processor.generate_summary_report(all_results)
            
            summary_path = Path(args.output) / f"summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            logger.info(f"Summary report saved to {summary_path}")
            logger.info(f"Processed {summary['successful']}/{summary['total_processed']} images successfully")
            logger.info(f"Average risk score: {summary['risk_statistics']['mean_risk_score']:.3f}")
            logger.info(f"SLA compliance rate: {summary['performance_statistics']['sla_compliance_rate']:.1%}")
        
        logger.info("Batch inference completed successfully")
        
    except Exception as e:
        logger.error(f"Batch inference failed: {e}")
        raise

if __name__ == "__main__":
    main()
