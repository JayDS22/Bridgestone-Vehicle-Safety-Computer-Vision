"""
Unit tests for vehicle safety models
"""

import pytest
import numpy as np
import pandas as pd
import cv2
import sys
from pathlib import Path
import tempfile
import os

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.yolo_detector import VehicleDetector
from src.models.feature_extractor import AdvancedFeatureExtractor
from src.models.ensemble_model import EnsembleRiskModel, NeuralNetworkClassifier
from src.models.survival_analysis import VehicleSafetySurvivalAnalysis

class TestVehicleDetector:
    """Test cases for VehicleDetector"""
    
    @pytest.fixture
    def detector(self):
        """Create detector instance for testing"""
        # Use a lightweight model for testing
        return VehicleDetector(
            model_path="yolov7n.pt",  # Nano version for faster testing
            confidence_threshold=0.5,
            device="cpu"
        )
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample test image"""
        return np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    def test_detector_initialization(self, detector):
        """Test detector initialization"""
        assert detector.confidence_threshold == 0.5
        assert detector.device in ["cpu", "cuda"]
        assert detector.input_size == (640, 640)
    
    def test_detect_single_image(self, detector, sample_image):
        """Test single image detection"""
        result = detector.detect(sample_image)
        
        assert 'detections' in result
        assert 'inference_time_ms' in result
        assert 'model_info' in result
        
        detections = result['detections']
        assert 'boxes' in detections
        assert 'scores' in detections
        assert 'classes' in detections
        assert 'labels' in detections
        
        # Check that inference time is reasonable
        assert 0 < result['inference_time_ms'] < 5000  # Max 5 seconds
    
    def test_detect_with_crops(self, detector, sample_image):
        """Test detection with crop extraction"""
        result = detector.detect(sample_image, return_crops=True)
        
        assert 'crops' in result
        assert isinstance(result['crops'], list)
    
    def test_annotate_frame(self, detector, sample_image):
        """Test frame annotation"""
        # Mock detections
        detections = {
            'boxes': [[100, 100, 200, 200], [300, 300, 400, 400]],
            'scores': [0.85, 0.92],
            'labels': ['car', 'truck']
        }
        
        annotated = detector.annotate_frame(sample_image, detections)
        
        assert annotated.shape == sample_image.shape
        assert annotated.dtype == sample_image.dtype
    
    def test_performance_stats(self, detector, sample_image):
        """Test performance statistics tracking"""
        # Run a few detections
        for _ in range(3):
            detector.detect(sample_image)
        
        stats = detector.get_performance_stats()
        
        assert 'mean_inference_time_ms' in stats
        assert 'throughput_fps' in stats
        assert stats['total_inferences'] == 3
    
    def test_benchmark(self, detector):
        """Test benchmark functionality"""
        results = detector.benchmark(num_iterations=10)
        
        assert 'mean_inference_time_ms' in results
        assert 'average_throughput_fps' in results
        assert results['total_benchmark_time_s'] > 0


class TestAdvancedFeatureExtractor:
    """Test cases for AdvancedFeatureExtractor"""
    
    @pytest.fixture
    def feature_extractor(self):
        """Create feature extractor instance"""
        return AdvancedFeatureExtractor(device="cpu")
    
    @pytest.fixture
    def sample_frame(self):
        """Create sample frame"""
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    @pytest.fixture
    def sample_detections(self):
        """Create sample detections"""
        return {
            'boxes': [[100, 100, 200, 200], [300, 150, 400, 250]],
            'scores': [0.85, 0.92],
            'labels': ['car', 'truck']
        }
    
    def test_feature_extraction(self, feature_extractor, sample_frame, sample_detections):
        """Test feature extraction"""
        features = feature_extractor.extract_features(sample_frame, sample_detections)
        
        assert isinstance(features, dict)
        assert len(features) > 50  # Should extract many features
        
        # Check for expected feature categories
        feature_names = list(features.keys())
        assert any('cnn' in name for name in feature_names)
        assert any('motion' in name for name in feature_names)
        assert any('spatial' in name for name in feature_names)
    
    def test_cnn_features(self, feature_extractor, sample_frame):
        """Test CNN feature extraction"""
        cnn_features = feature_extractor._extract_cnn_features(sample_frame)
        
        assert isinstance(cnn_features, dict)
        assert 'cnn_mean' in cnn_features
        assert 'cnn_std' in cnn_features
        assert len(cnn_features) == 9  # Expected number of CNN summary features
    
    def test_spatial_features(self, feature_extractor, sample_detections):
        """Test spatial feature extraction"""
        frame_shape = (480, 640, 3)
        spatial_features = feature_extractor._extract_spatial_features(sample_detections, frame_shape)
        
        assert isinstance(spatial_features, dict)
        assert 'num_detections' in spatial_features
        assert 'detection_density' in spatial_features
        assert spatial_features['num_detections'] == 2
    
    def test_temporal_features(self, feature_extractor, sample_detections):
        """Test temporal feature extraction"""
        # First call (no previous detections)
        temporal_features1 = feature_extractor._extract_temporal_features(sample_detections)
        assert isinstance(temporal_features1, dict)
        
        # Second call (with previous detections)
        temporal_features2 = feature_extractor._extract_temporal_features(sample_detections)
        assert 'mean_object_persistence' in temporal_features2


class TestEnsembleRiskModel:
    """Test cases for EnsembleRiskModel"""
    
    @pytest.fixture
    def ensemble_model(self):
        """Create ensemble model instance"""
        return EnsembleRiskModel(device="cpu")
    
    @pytest.fixture
    def sample_training_data(self):
        """Create sample training data"""
        np.random.seed(42)
        n_samples = 1000
        n_features = 50
        
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        y = pd.Series((X.iloc[:, 0] + X.iloc[:, 1] + np.random.randn(n_samples) * 0.1 > 0).astype(int))
        
        return X, y
    
    def test_model_initialization(self, ensemble_model):
        """Test model initialization"""
        assert not ensemble_model.is_trained
        assert ensemble_model.model_weights['random_forest'] == 0.3
        assert ensemble_model.model_weights['xgboost'] == 0.4
        assert ensemble_model.model_weights['neural_network'] == 0.3
    
    def test_training(self, ensemble_model, sample_training_data):
        """Test model training"""
        X, y = sample_training_data
        
        # Train with small number of epochs for speed
        metrics = ensemble_model.train(X, y, epochs=5, batch_size=64)
        
        assert ensemble_model.is_trained
        assert 'ensemble_auc' in metrics
        assert 'ensemble_accuracy' in metrics
        assert metrics['ensemble_auc'] > 0.5  # Should be better than random
    
    def test_prediction(self, ensemble_model, sample_training_data):
        """Test model predictions"""
        X, y = sample_training_data
        
        # Train model first
        ensemble_model.train(X, y, epochs=5, batch_size=64)
        
        # Test predictions
        predictions = ensemble_model.predict(X.head(10))
        assert len(predictions) == 10
        assert all(pred in [0, 1] for pred in predictions)
        
        # Test probability predictions
        probabilities = ensemble_model.predict_proba(X.head(10))
        assert probabilities.shape == (10, 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0)
        
        # Test risk scores
        risk_scores = ensemble_model.predict_risk_scores(X.head(10))
        assert len(risk_scores) == 10
        assert all(0 <= score <= 1 for score in risk_scores)
    
    def test_model_save_load(self, ensemble_model, sample_training_data):
        """Test model saving and loading"""
        X, y = sample_training_data
        
        # Train model
        ensemble_model.train(X, y, epochs=5, batch_size=64)
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            model_path = f.name
        
        try:
            # Save model
            ensemble_model.save_model(model_path)
            assert os.path.exists(model_path)
            
            # Load model
            new_model = EnsembleRiskModel()
            new_model.load_model(model_path)
            
            assert new_model.is_trained
            assert new_model.feature_names == ensemble_model.feature_names
            
            # Test that predictions are the same
            original_pred = ensemble_model.predict_risk_scores(X.head(5))
            loaded_pred = new_model.predict_risk_scores(X.head(5))
            
            np.testing.assert_array_almost_equal(original_pred, loaded_pred, decimal=5)
        
        finally:
            if os.path.exists(model_path):
                os.unlink(model_path)


class TestNeuralNetworkClassifier:
    """Test cases for NeuralNetworkClassifier"""
    
    def test_network_initialization(self):
        """Test neural network initialization"""
        input_dim = 100
        hidden_dims = [256, 128, 64]
        
        network = NeuralNetworkClassifier(input_dim, hidden_dims)
        
        # Test forward pass
        x = torch.randn(32, input_dim)
        output = network(x)
        
        assert output.shape == (32, 1)
        assert torch.all(output >= 0) and torch.all(output <= 1)  # Sigmoid output


class TestVehicleSafetySurvivalAnalysis:
    """Test cases for VehicleSafetySurvivalAnalysis"""
    
    @pytest.fixture
    def survival_model(self):
        """Create survival analysis model"""
        return VehicleSafetySurvivalAnalysis(penalizer=0.1)
    
    @pytest.fixture
    def sample_survival_data(self):
        """Create sample survival data"""
        np.random.seed(42)
        n_samples = 500
        
        # Generate synthetic survival data
        data = pd.DataFrame({
            'vehicle_age': np.random.exponential(5, n_samples),
            'driver_age': np.random.normal(40, 15, n_samples),
            'speed_violations': np.random.poisson(2, n_samples),
            'weather_score': np.random.uniform(0, 1, n_samples),
            'road_quality': np.random.uniform(0, 1, n_samples)
        })
        
        # Generate survival times based on covariates
        risk_score = (data['vehicle_age'] * 0.1 + 
                     data['speed_violations'] * 0.3 + 
                     (1 - data['weather_score']) * 0.2)
        
        survival_times = np.random.exponential(1 / (0.1 + risk_score * 0.05))
        survival_times = np.clip(survival_times, 0.1, 60)  # 0.1 to 60 months
        
        events = np.random.binomial(1, 0.3, n_samples)  # 30% event rate
        
        data['time_to_crash'] = survival_times
        data['crash_occurred'] = events
        
        return data
    
    def test_data_preparation(self, survival_model, sample_survival_data):
        """Test survival data preparation"""
        prepared_data = survival_model.prepare_survival_data(
            sample_survival_data,
            duration_col='time_to_crash',
            event_col='crash_occurred'
        )
        
        assert len(prepared_data) <= len(sample_survival_data)  # May filter some data
        assert 'time_to_crash' in prepared_data.columns
        assert 'crash_occurred' in prepared_data.columns
    
    def test_model_training(self, survival_model, sample_survival_data):
        """Test survival model training"""
        prepared_data = survival_model.prepare_survival_data(sample_survival_data)
        
        metrics = survival_model.fit(prepared_data, validation_split=0.2)
        
        assert survival_model.is_fitted
        assert 'val_c_index' in metrics
        assert 'train_c_index' in metrics
        assert 0.5 <= metrics['val_c_index'] <= 1.0  # C-index should be reasonable
    
    def test_risk_prediction(self, survival_model, sample_survival_data):
        """Test risk score prediction"""
        prepared_data = survival_model.prepare_survival_data(sample_survival_data)
        survival_model.fit(prepared_data, validation_split=0.2)
        
        # Test risk scores
        test_features = prepared_data[survival_model.feature_names].head(10)
        risk_scores = survival_model.predict_risk_scores(test_features)
        
        assert len(risk_scores) == 10
        assert all(score > 0 for score in risk_scores)
    
    def test_survival_probabilities(self, survival_model, sample_survival_data):
        """Test survival probability prediction"""
        prepared_data = survival_model.prepare_survival_data(sample_survival_data)
        survival_model.fit(prepared_data, validation_split=0.2)
        
        test_features = prepared_data[survival_model.feature_names].head(5)
        survival_probs = survival_model.predict_survival_probabilities(
            test_features, time_points=[1, 6, 12]
        )
        
        assert survival_probs.shape == (5, 3)
        assert all(0 <= prob <= 1 for prob in survival_probs.values.flatten())
    
    def test_risk_stratification(self, survival_model, sample_survival_data):
        """Test risk stratification"""
        prepared_data = survival_model.prepare_survival_data(sample_survival_data)
        survival_model.fit(prepared_data, validation_split=0.2)
        
        test_features = prepared_data[survival_model.feature_names].head(20)
        risk_strata = survival_model.stratify_risk(test_features)
        
        assert 'risk_score' in risk_strata.columns
        assert 'risk_category' in risk_strata.columns
        assert set(risk_strata['risk_category'].unique()).issubset({'low', 'medium', 'high'})


# Integration test
class TestModelIntegration:
    """Test integration between different models"""
    
    def test_full_pipeline(self):
        """Test the complete inference pipeline"""
        # Create sample data
        sample_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # Initialize models (with CPU/lightweight settings for testing)
        detector = VehicleDetector(model_path="yolov7n.pt", device="cpu")
        feature_extractor = AdvancedFeatureExtractor(device="cpu")
        
        # Run detection
        detection_result = detector.detect(sample_image)
        
        # Extract features
        features = feature_extractor.extract_features(
            sample_image, 
            detection_result['detections']
        )
        
        # Verify pipeline works end-to-end
        assert isinstance(detection_result, dict)
        assert isinstance(features, dict)
        assert len(features) > 10  # Should extract meaningful number of features
        
        # Verify feature values are reasonable
        for key, value in features.items():
            assert isinstance(value, (int, float))
            assert not np.isnan(value)
            assert not np.isinf(value)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
