"""
AWS Lambda function for vehicle safety inference
Serverless deployment of the Bridgestone vehicle safety system
"""

import json
import base64
import boto3
import numpy as np
import cv2
import logging
import os
from typing import Dict, Any
import time

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Global variables for model caching
models = {}
s3_client = None

def initialize_models():
    """Initialize models on cold start"""
    global models, s3_client
    
    if models:
        return  # Already initialized
    
    try:
        logger.info("Initializing models...")
        
        # Initialize S3 client
        s3_client = boto3.client('s3')
        
        # Download models from S3 if not already cached
        model_bucket = os.environ.get('MODEL_BUCKET', 'bridgestone-vehicle-safety')
        model_files = {
            'yolo': 'models/yolov7_vehicle.pt',
            'ensemble': 'models/ensemble_model.pkl',
            'survival': 'models/survival_model.pkl'
        }
        
        local_model_dir = '/tmp/models/'
        os.makedirs(local_model_dir, exist_ok=True)
        
        # Download models
        for model_name, s3_key in model_files.items():
            local_path = os.path.join(local_model_dir, os.path.basename(s3_key))
            
            if not os.path.exists(local_path):
                logger.info(f"Downloading {model_name} model from S3...")
                s3_client.download_file(model_bucket, s3_key, local_path)
            
            # Store local path
            models[f'{model_name}_path'] = local_path
        
        # Import and initialize model classes
        # Note: In production, you'd need to include these dependencies in Lambda layer
        from ultralytics import YOLO
        import pickle
        
        # Initialize YOLO detector
        models['detector'] = YOLO(models['yolo_path'])
        
        # Load ensemble model
        with open(models['ensemble_path'], 'rb') as f:
            models['ensemble'] = pickle.load(f)
        
        # Load survival model
        with open(models['survival_path'], 'rb') as f:
            models['survival'] = pickle.load(f)
        
        logger.info("Models initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}")
        raise

def preprocess_image(image_b64: str) -> np.ndarray:
    """
    Preprocess base64 encoded image
    
    Args:
        image_b64: Base64 encoded image
        
    Returns:
        Preprocessed image array
    """
    try:
        # Decode base64 image
        image_data = base64.b64decode(image_b64)
        image_array = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Invalid image data")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image
        
    except Exception as e:
        logger.error(f"Image preprocessing failed: {e}")
        raise

def run_inference(image: np.ndarray, metadata: Dict = None) -> Dict[str, Any]:
    """
    Run complete inference pipeline
    
    Args:
        image: Preprocessed image
        metadata: Optional metadata
        
    Returns:
        Inference results
    """
    start_time = time.time()
    
    try:
        # Step 1: Vehicle Detection
        detection_results = models['detector'].predict(image, verbose=False)
        
        # Extract detection data
        detections = {
            'boxes': [],
            'scores': [],
            'labels': []
        }
        
        if detection_results and len(detection_results) > 0:
            result = detection_results[0]
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)
                
                detections['boxes'] = boxes.tolist()
                detections['scores'] = scores.tolist()
                detections['labels'] = [result.names[cls] for cls in classes]
        
        # Step 2: Feature Extraction (simplified for Lambda)
        # In production, you'd include the full feature extractor
        basic_features = extract_basic_features(image, detections)
        
        # Step 3: Risk Assessment
        risk_score = models['ensemble'].predict_risk_scores(
            np.array(basic_features).reshape(1, -1)
        )[0] if basic_features else 0.5
        
        # Step 4: Crash Probability (simplified)
        crash_probabilities = {
            '1m': float(risk_score * 0.1),
            '3m': float(risk_score * 0.2), 
            '6m': float(risk_score * 0.35),
            '12m': float(risk_score * 0.5)
        }
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        
        return {
            'predictions': {
                'vehicles': detections,
                'risk_score': float(risk_score),
                'crash_probability': crash_probabilities,
                'detection_count': len(detections['boxes']),
                'confidence_scores': detections['scores']
            },
            'processing_time': processing_time,
            'model_version': '2.1.0',
            'timestamp': time.time()
        }
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise

def extract_basic_features(image: np.ndarray, detections: Dict) -> list:
    """
    Extract basic features for risk assessment
    Simplified version for Lambda deployment
    
    Args:
        image: Input image
        detections: Detection results
        
    Returns:
        List of basic features
    """
    try:
        features = []
        
        # Basic image statistics
        features.extend([
            float(np.mean(image)),          # Mean brightness
            float(np.std(image)),           # Brightness variation
            float(np.mean(image[:,:,0])),   # Red channel mean
            float(np.mean(image[:,:,1])),   # Green channel mean
            float(np.mean(image[:,:,2]))    # Blue channel mean
        ])
        
        # Detection-based features
        num_detections = len(detections.get('boxes', []))
        avg_confidence = np.mean(detections.get('scores', [0])) if detections.get('scores') else 0
        
        features.extend([
            float(num_detections),
            float(avg_confidence),
            float(num_detections > 5),  # High density indicator
            float(avg_confidence > 0.8) # High confidence indicator
        ])
        
        # Pad features to expected size (simplified feature set)
        while len(features) < 20:
            features.append(0.0)
        
        return features[:20]  # Return first 20 features
        
    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        return [0.0] * 20

def lambda_handler(event, context):
    """
    AWS Lambda handler function
    
    Args:
        event: Lambda event object
        context: Lambda context object
        
    Returns:
        API Gateway response
    """
    try:
        # Initialize models on cold start
        initialize_models()
        
        # Parse request
        if 'body' in event:
            if isinstance(event['body'], str):
                body = json.loads(event['body'])
            else:
                body = event['body']
        else:
            body = event
        
        # Validate input
        if 'image' not in body:
            return {
                'statusCode': 400,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({
                    'error': 'Missing required field: image'
                })
            }
        
        # Process image
        image = preprocess_image(body['image'])
        metadata = body.get('metadata', {})
        
        # Run inference
        results = run_inference(image, metadata)
        
        # Log performance
        logger.info(f"Inference completed in {results['processing_time']:.1f}ms")
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps(results, default=str)
        }
        
    except Exception as e:
        logger.error(f"Lambda execution failed: {e}")
        
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'error': str(e),
                'timestamp': time.time()
            })
        }

def health_check_handler(event, context):
    """
    Health check handler for Lambda function
    
    Args:
        event: Lambda event object
        context: Lambda context object
        
    Returns:
        Health check response
    """
    try:
        # Check if models are initialized
        models_loaded = bool(models)
        
        # Basic memory check
        import psutil
        memory_percent = psutil.virtual_memory().percent
        
        health_status = {
            'status': 'healthy' if models_loaded and memory_percent < 90 else 'unhealthy',
            'models_loaded': models_loaded,
            'memory_usage_percent': memory_percent,
            'lambda_version': context.function_version if context else 'unknown',
            'timestamp': time.time()
        }
        
        status_code = 200 if health_status['status'] == 'healthy' else 503
        
        return {
            'statusCode': status_code,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps(health_status)
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': time.time()
            })
        }

# For local testing
if __name__ == "__main__":
    # Test event
    test_event = {
        'body': json.dumps({
            'image': 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==',
            'metadata': {'test': True}
        })
    }
    
    # Mock context
    class MockContext:
        function_version = '$LATEST'
    
    # Run test
    response = lambda_handler(test_event, MockContext())
    print(json.dumps(response, indent=2))
