"""
API testing for Bridgestone Vehicle Safety System
Tests FastAPI endpoints and functionality
"""

import pytest
import asyncio
import json
import base64
import numpy as np
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import sys
from pathlib import Path
import tempfile
import cv2

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Mock the models before importing the API
with patch('src.models.yolo_detector.VehicleDetector'), \
     patch('src.models.feature_extractor.AdvancedFeatureExtractor'), \
     patch('src.models.ensemble_model.EnsembleRiskModel'), \
     patch('src.models.survival_analysis.VehicleSafetySurvivalAnalysis'):
    from src.api.inference_api import app

client = TestClient(app)

class TestAPIEndpoints:
    """Test API endpoints"""
    
    @pytest.fixture
    def sample_image_b64(self):
        """Create a sample base64 encoded image"""
        # Create a simple test image
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Encode as JPEG
        _, buffer = cv2.imencode('.jpg', image)
        
        # Convert to base64
        image_b64 = base64.b64encode(buffer).decode('utf-8')
        
        return image_b64
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "models_loaded" in data
        assert "version" in data
    
    def test_predict_endpoint_success(self, sample_image_b64):
        """Test successful prediction"""
        request_data = {
            "image": sample_image_b64,
            "metadata": {"test": True}
        }
        
        with patch('src.api.inference_api.models') as mock_models:
            # Mock model responses
            mock_detector = Mock()
            mock_detector.detect.return_value = {
                "detections": {
                    "boxes": [[100, 100, 200, 200]],
                    "scores": [0.85],
                    "labels": ["car"]
                },
                "inference_time_ms": 125.5
            }
            
            mock_feature_extractor = Mock()
            mock_feature_extractor.extract_features.return_value = {
                "feature_1": 0.5,
                "feature_2": 0.8
            }
            
            mock_ensemble = Mock()
            mock_ensemble.predict_risk_scores.return_value = [0.75]
            
            mock_survival = Mock()
            mock_survival.predict_crash_probabilities.return_value = Mock()
            mock_survival.predict_crash_probabilities.return_value.iloc = [
                Mock(to_dict=lambda: {"1m": 0.1, "3m": 0.2, "6m": 0.35, "12m": 0.5})
            ]
            
            mock_models["detector"] = mock_detector
            mock_models["feature_extractor"] = mock_feature_extractor
            mock_models["ensemble"] = mock_ensemble
            mock_models["survival"] = mock_survival
            mock_models["performance_monitor"] = Mock()
            
            response = client.post("/predict", json=request_data)
        
        assert response.status_code == 200
        
        data = response.json()
        assert "predictions" in data
        assert "processing_time" in data
        assert "model_version" in data
        assert "timestamp" in data
        
        predictions = data["predictions"]
        assert "vehicles" in predictions
        assert "risk_score" in predictions
        assert "crash_probability" in predictions
        assert predictions["risk_score"] == 0.75
    
    def test_predict_endpoint_missing_image(self):
        """Test prediction with missing image"""
        request_data = {"metadata": {"test": True}}
        
        response = client.post("/predict", json=request_data)
        assert response.status_code == 400
        
        data = response.json()
        assert "detail" in data
    
    def test_predict_endpoint_invalid_image(self):
        """Test prediction with invalid image data"""
        request_data = {
            "image": "invalid_base64_data",
            "metadata": {"test": True}
        }
        
        response = client.post("/predict", json=request_data)
        assert response.status_code == 500
    
    def test_predict_batch_endpoint(self, sample_image_b64):
        """Test batch prediction endpoint"""
        request_data = {
            "images": [sample_image_b64, sample_image_b64],
            "metadata": [{"test": True}, {"test": True}]
        }
        
        with patch('src.api.inference_api.models') as mock_models:
            # Mock model responses
            mock_detector = Mock()
            mock_detector.detect.return_value = {
                "detections": {
                    "boxes": [[100, 100, 200, 200]],
                    "scores": [0.85],
                    "labels": ["car"]
                },
                "inference_time_ms": 125.5
            }
            
            mock_feature_extractor = Mock()
            mock_feature_extractor.extract_features.return_value = {
                "feature_1": 0.5,
                "feature_2": 0.8
            }
            
            mock_ensemble = Mock()
            mock_ensemble.predict_risk_scores.return_value = [0.75]
            
            mock_models["detector"] = mock_detector
            mock_models["feature_extractor"] = mock_feature_extractor
            mock_models["ensemble"] = mock_ensemble
            
            response = client.post("/predict/batch", json=request_data)
        
        assert response.status_code == 200
        
        data = response.json()
        assert "results" in data
        assert "total_processing_time" in data
        assert "batch_size" in data
        assert len(data["results"]) == 2
    
    def test_predict_video_endpoint(self):
        """Test video prediction endpoint"""
        request_data = {
            "video_path": "/tmp/test_video.mp4",
            "frame_skip": 1,
            "max_frames": 10
        }
        
        with patch('src.api.inference_api.models') as mock_models, \
             patch('os.path.exists') as mock_exists:
            
            mock_exists.return_value = True
            
            # Mock model responses
            mock_detector = Mock()
            mock_detector.detect_video.return_value = [
                {
                    "frame_number": 1,
                    "timestamp": 0.033,
                    "detections": {
                        "boxes": [[100, 100, 200, 200]],
                        "scores": [0.85],
                        "labels": ["car"]
                    },
                    "inference_time_ms": 125.5
                }
            ]
            
            mock_feature_extractor = Mock()
            mock_ensemble = Mock()
            
            mock_models["detector"] = mock_detector
            mock_models["feature_extractor"] = mock_feature_extractor
            mock_models["ensemble"] = mock_ensemble
            
            response = client.post("/predict/video", json=request_data)
        
        assert response.status_code == 200
        
        data = response.json()
        assert "job_id" in data
        assert "status" in data
        assert data["status"] == "processing"
    
    def test_metrics_endpoint(self):
        """Test metrics endpoint"""
        with patch('src.api.inference_api.models') as mock_models:
            mock_performance_monitor = Mock()
            mock_performance_monitor.get_metrics.return_value = {
                "system": {"uptime_seconds": 100},
                "requests": {"total": 10, "successful": 9, "failed": 1}
            }
            mock_models["performance_monitor"] = mock_performance_monitor
            
            response = client.get("/metrics")
        
        assert response.status_code == 200
        
        data = response.json()
        assert "system" in data
        assert "requests" in data
    
    def test_models_info_endpoint(self):
        """Test models info endpoint"""
        with patch('src.api.inference_api.models') as mock_models:
            mock_detector = Mock()
            mock_detector.get_performance_stats.return_value = {
                "mean_inference_time_ms": 125.5
            }
            mock_detector.VEHICLE_CLASSES = {"car": 2, "truck": 7}
            
            mock_ensemble = Mock()
            mock_ensemble.get_model_info.return_value = {
                "is_trained": True,
                "model_weights": {"rf": 0.3, "xgb": 0.4, "nn": 0.3}
            }
            
            mock_survival = Mock()
            mock_survival.is_fitted = True
            mock_survival.training_metrics = {"c_index": 0.78}
            
            mock_models["detector"] = mock_detector
            mock_models["ensemble"] = mock_ensemble
            mock_models["survival"] = mock_survival
            
            response = client.get("/models/info")
        
        assert response.status_code == 200
        
        data = response.json()
        assert "detector" in data
        assert "ensemble" in data
        assert "survival" in data
    
    def test_benchmark_endpoint(self):
        """Test benchmark endpoint"""
        with patch('src.api.inference_api.models') as mock_models:
            mock_detector = Mock()
            mock_detector.benchmark.return_value = {
                "mean_inference_time_ms": 125.5,
                "average_throughput_fps": 8.0
            }
            mock_models["detector"] = mock_detector
            
            response = client.post("/benchmark?iterations=10")
        
        assert response.status_code == 200
        
        data = response.json()
        assert "benchmark_results" in data
        assert "timestamp" in data
    
    def test_upload_endpoint(self, sample_image_b64):
        """Test file upload endpoint"""
        # Create a temporary image file
        image_data = base64.b64decode(sample_image_b64)
        
        with patch('src.api.inference_api.models') as mock_models:
            # Mock model responses
            mock_detector = Mock()
            mock_detector.detect.return_value = {
                "detections": {
                    "boxes": [[100, 100, 200, 200]],
                    "scores": [0.85],
                    "labels": ["car"]
                }
            }
            
            mock_feature_extractor = Mock()
            mock_feature_extractor.extract_features.return_value = {
                "feature_1": 0.5
            }
            
            mock_ensemble = Mock()
            mock_ensemble.predict_risk_scores.return_value = [0.75]
            
            mock_models["detector"] = mock_detector
            mock_models["feature_extractor"] = mock_feature_extractor
            mock_models["ensemble"] = mock_ensemble
            
            files = {"file": ("test.jpg", image_data, "image/jpeg")}
            response = client.post("/upload", files=files)
        
        assert response.status_code == 200
        
        data = response.json()
        assert "filename" in data
        assert "risk_score" in data
        assert "detections" in data


class TestAPIErrorHandling:
    """Test API error handling"""
    
    def test_404_handler(self):
        """Test 404 error handler"""
        response = client.get("/nonexistent-endpoint")
        assert response.status_code == 404
        
        data = response.json()
        assert "detail" in data
    
    def test_500_handler(self):
        """Test 500 error handler by causing an internal error"""
        with patch('src.api.inference_api.models', side_effect=Exception("Test error")):
            response = client.get("/metrics")
        
        assert response.status_code == 500


class TestAPIPerformance:
    """Test API performance characteristics"""
    
    def test_response_time(self, sample_image_b64):
        """Test that response times meet SLA"""
        import time
        
        request_data = {
            "image": sample_image_b64,
            "metadata": {"test": True}
        }
        
        with patch('src.api.inference_api.models') as mock_models:
            # Mock fast responses
            mock_detector = Mock()
            mock_detector.detect.return_value = {
                "detections": {"boxes": [], "scores": [], "labels": []},
                "inference_time_ms": 100
            }
            
            mock_feature_extractor = Mock()
            mock_feature_extractor.extract_features.return_value = {"feature_1": 0.5}
            
            mock_ensemble = Mock()
            mock_ensemble.predict_risk_scores.return_value = [0.5]
            
            mock_survival = Mock()
            mock_survival.predict_crash_probabilities.return_value = Mock()
            mock_survival.predict_crash_probabilities.return_value.iloc = [
                Mock(to_dict=lambda: {"1m": 0.1})
            ]
            
            mock_models["detector"] = mock_detector
            mock_models["feature_extractor"] = mock_feature_extractor
            mock_models["ensemble"] = mock_ensemble
            mock_models["survival"] = mock_survival
            mock_models["performance_monitor"] = Mock()
            
            start_time = time.time()
            response = client.post("/predict", json=request_data)
            end_time = time.time()
        
        assert response.status_code == 200
        
        # Response should be under 1 second (accounting for test overhead)
        response_time = end_time - start_time
        assert response_time < 1.0
        
        # Check that processing time is reported
        data = response.json()
        assert data["processing_time"] < 1000  # Should be under 1000ms
    
    def test_concurrent_requests(self, sample_image_b64):
        """Test handling of concurrent requests"""
        import concurrent.futures
        import threading
        
        request_data = {
            "image": sample_image_b64,
            "metadata": {"test": True}
        }
        
        def make_request():
            with patch('src.api.inference_api.models') as mock_models:
                # Mock responses (simplified for concurrency test)
                mock_detector = Mock()
                mock_detector.detect.return_value = {
                    "detections": {"boxes": [], "scores": [], "labels": []},
                    "inference_time_ms": 100
                }
                
                mock_feature_extractor = Mock()
                mock_feature_extractor.extract_features.return_value = {"feature_1": 0.5}
                
                mock_ensemble = Mock()
                mock_ensemble.predict_risk_scores.return_value = [0.5]
                
                mock_survival = Mock()
                mock_survival.predict_crash_probabilities.return_value = Mock()
                mock_survival.predict_crash_probabilities.return_value.iloc = [
                    Mock(to_dict=lambda: {"1m": 0.1})
                ]
                
                mock_models["detector"] = mock_detector
                mock_models["feature_extractor"] = mock_feature_extractor
                mock_models["ensemble"] = mock_ensemble
                mock_models["survival"] = mock_survival
                mock_models["performance_monitor"] = Mock()
                
                return client.post("/predict", json=request_data)
        
        # Make 5 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(5)]
            responses = [future.result() for future in futures]
        
        # All requests should succeed
        assert all(response.status_code == 200 for response in responses)


class TestAPIValidation:
    """Test API input validation"""
    
    def test_predict_input_validation(self):
        """Test input validation for predict endpoint"""
        # Test with missing required fields
        response = client.post("/predict", json={})
        assert response.status_code == 400
        
        # Test with invalid data types
        response = client.post("/predict", json={"image": 123})
        assert response.status_code == 422  # Validation error
    
    def test_batch_input_validation(self):
        """Test input validation for batch endpoint"""
        # Test with empty images list
        response = client.post("/predict/batch", json={"images": []})
        assert response.status_code == 200  # Should handle empty list gracefully
        
        # Test with mismatched metadata length
        response = client.post("/predict/batch", json={
            "images": ["img1", "img2"],
            "metadata": [{"test": True}]  # Only one metadata for two images
        })
        assert response.status_code == 200  # Should handle gracefully


# Integration tests
class TestAPIIntegration:
    """Integration tests for the complete API"""
    
    @pytest.mark.asyncio
    async def test_full_inference_pipeline(self, sample_image_b64):
        """Test the complete inference pipeline through the API"""
        request_data = {
            "image": sample_image_b64,
            "metadata": {
                "location": {"lat": 40.7128, "lon": -74.0060},
                "timestamp": "2024-01-01T12:00:00Z",
                "weather": "clear"
            }
        }
        
        with patch('src.api.inference_api.models') as mock_models:
            # Mock complete pipeline
            mock_detector = Mock()
            mock_detector.detect.return_value = {
                "detections": {
                    "boxes": [[100, 100, 200, 200], [300, 150, 400, 250]],
                    "scores": [0.85, 0.92],
                    "labels": ["car", "truck"]
                },
                "inference_time_ms": 125.5
            }
            
            mock_feature_extractor = Mock()
            mock_feature_extractor.extract_features.return_value = {
                f"feature_{i}": np.random.random() for i in range(50)
            }
            
            mock_ensemble = Mock()
            mock_ensemble.predict_risk_scores.return_value = [0.75]
            
            mock_survival = Mock()
            mock_survival.predict_crash_probabilities.return_value = Mock()
            mock_survival.predict_crash_probabilities.return_value.iloc = [
                Mock(to_dict=lambda: {
                    "crash_prob_1m": 0.02,
                    "crash_prob_3m": 0.08,
                    "crash_prob_6m": 0.15,
                    "crash_prob_12m": 0.28
                })
            ]
            
            mock_models["detector"] = mock_detector
            mock_models["feature_extractor"] = mock_feature_extractor
            mock_models["ensemble"] = mock_ensemble
            mock_models["survival"] = mock_survival
            mock_models["performance_monitor"] = Mock()
            
            response = client.post("/predict", json=request_data)
        
        assert response.status_code == 200
        
        data = response.json()
        
        # Verify complete response structure
        assert "predictions" in data
        predictions = data["predictions"]
        
        assert "vehicles" in predictions
        assert len(predictions["vehicles"]["boxes"]) == 2
        assert len(predictions["vehicles"]["labels"]) == 2
        
        assert "risk_score" in predictions
        assert 0 <= predictions["risk_score"] <= 1
        
        assert "crash_probability" in predictions
        crash_probs = predictions["crash_probability"]
        assert "1m" in crash_probs
        assert "3m" in crash_probs
        assert "6m" in crash_probs
        assert "12m" in crash_probs
        
        assert "detection_count" in predictions
        assert predictions["detection_count"] == 2
        
        assert "processing_time" in data
        assert "model_version" in data
        assert "timestamp" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
