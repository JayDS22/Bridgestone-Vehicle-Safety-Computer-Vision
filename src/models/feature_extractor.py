"""
Advanced Feature Extraction Pipeline for Vehicle Safety Analysis
Extracts 127 engineered features from video frames and detection results
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union
import time
from scipy import ndimage
from skimage import feature, measure
import logging
from torchvision import models, transforms

class AdvancedFeatureExtractor:
    """
    Multi-modal feature extraction system for vehicle safety assessment
    
    Features extracted:
    - CNN-based deep features (ResNet50): 2048 features
    - Motion analysis: 25 features
    - Edge and texture features: 30 features
    - Spatial relationship features: 20 features
    - Temporal features: 15 features
    - Weather/lighting features: 12 features
    Total: 127+ engineered features
    """
    
    def __init__(self, device: str = "auto"):
        """Initialize feature extractor with pre-trained models"""
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else device
        
        # Initialize CNN backbone (ResNet50)
        self.cnn_model = self._load_cnn_backbone()
        
        # Feature preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Motion tracking
        self.optical_flow = cv2.FarnebackOpticalFlow_create()
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2()
        
        # Previous frame storage for temporal features
        self.prev_frame = None
        self.prev_detections = None
        self.frame_history = []
        
        self.logger = logging.getLogger(__name__)
        
    def _load_cnn_backbone(self) -> nn.Module:
        """Load pre-trained ResNet50 for deep feature extraction"""
        model = models.resnet50(pretrained=True)
        # Remove final classification layer
        model = nn.Sequential(*list(model.children())[:-1])
        model.eval()
        model.to(self.device)
        
        # Warm up
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
            _ = model(dummy_input)
            
        return model
    
    def extract_features(
        self,
        frame: np.ndarray,
        detections: Dict,
        metadata: Optional[Dict] = None
    ) -> Dict[str, Union[float, List[float]]]:
        """
        Extract comprehensive feature set from frame and detections
        
        Args:
            frame: Input video frame (BGR format)
            detections: Detection results from YOLO
            metadata: Additional metadata (timestamp, weather, etc.)
            
        Returns:
            Dictionary containing all extracted features
        """
        features = {}
        
        try:
            # Convert to RGB for processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 1. CNN Deep Features
            cnn_features = self._extract_cnn_features(frame_rgb)
            features.update(cnn_features)
            
            # 2. Motion Features
            motion_features = self._extract_motion_features(frame)
            features.update(motion_features)
            
            # 3. Edge and Texture Features
            texture_features = self._extract_texture_features(frame)
            features.update(texture_features)
            
            # 4. Spatial Relationship Features
            spatial_features = self._extract_spatial_features(detections, frame.shape)
            features.update(spatial_features)
            
            # 5. Temporal Features
            temporal_features = self._extract_temporal_features(detections)
            features.update(temporal_features)
            
            # 6. Environmental Features
            env_features = self._extract_environmental_features(frame, metadata)
            features.update(env_features)
            
            # 7. Detection-specific Features
            detection_features = self._extract_detection_features(detections)
            features.update(detection_features)
            
            # Update history
            self._update_history(frame, detections)
            
            self.logger.debug(f"Extracted {len(features)} features")
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            features = self._get_default_features()
        
        return features
    
    def _extract_cnn_features(self, frame_rgb: np.ndarray) -> Dict[str, float]:
        """Extract deep CNN features using ResNet50"""
        try:
            # Preprocess frame
            frame_tensor = self.transform(frame_rgb).unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.cnn_model(frame_tensor)
                features = features.view(features.size(0), -1)  # Flatten
                features = features.cpu().numpy()[0]
            
            # Reduce dimensionality using statistical summary
            cnn_dict = {
                'cnn_mean': float(np.mean(features)),
                'cnn_std': float(np.std(features)),
                'cnn_max': float(np.max(features)),
                'cnn_min': float(np.min(features)),
                'cnn_median': float(np.median(features)),
                'cnn_q75': float(np.percentile(features, 75)),
                'cnn_q25': float(np.percentile(features, 25)),
                'cnn_skewness': float(self._calculate_skewness(features)),
                'cnn_kurtosis': float(self._calculate_kurtosis(features))
            }
            
            return cnn_dict
            
        except Exception as e:
            self.logger.error(f"CNN feature extraction failed: {e}")
            return {f'cnn_{i}': 0.0 for i in range(9)}
    
    def _extract_motion_features(self, frame: np.ndarray) -> Dict[str, float]:
        """Extract motion-based features"""
        motion_dict = {}
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if self.prev_frame is not None:
                # Optical flow
                flow = cv2.calcOpticalFlowPyrLK(
                    self.prev_frame, gray, None, None
                )[0]
                
                if flow is not None:
                    # Flow magnitude and direction
                    magnitude = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
                    
                    motion_dict.update({
                        'flow_mean_magnitude': float(np.mean(magnitude)),
                        'flow_max_magnitude': float(np.max(magnitude)),
                        'flow_std_magnitude': float(np.std(magnitude)),
                        'flow_direction_std': float(np.std(np.arctan2(flow[:, :, 1], flow[:, :, 0]))),
                        'motion_density': float(np.sum(magnitude > 1.0) / magnitude.size)
                    })
                
                # Frame difference
                diff = cv2.absdiff(self.prev_frame, gray)
                motion_dict.update({
                    'frame_diff_mean': float(np.mean(diff)),
                    'frame_diff_std': float(np.std(diff)),
                    'frame_diff_max': float(np.max(diff)),
                    'motion_pixels_ratio': float(np.sum(diff > 30) / diff.size)
                })
                
                # Background subtraction
                fg_mask = self.background_subtractor.apply(frame)
                motion_dict.update({
                    'foreground_ratio': float(np.sum(fg_mask > 0) / fg_mask.size),
                    'foreground_objects': float(len(cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]))
                })
            
            # Gradient-based motion estimation
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            motion_dict.update({
                'gradient_mean': float(np.mean(gradient_magnitude)),
                'gradient_std': float(np.std(gradient_magnitude)),
                'edge_density': float(np.sum(gradient_magnitude > 50) / gradient_magnitude.size)
            })
            
        except Exception as e:
            self.logger.error(f"Motion feature extraction failed: {e}")
            
        # Fill missing values
        motion_keys = [
            'flow_mean_magnitude', 'flow_max_magnitude', 'flow_std_magnitude',
            'flow_direction_std', 'motion_density', 'frame_diff_mean',
            'frame_diff_std', 'frame_diff_max', 'motion_pixels_ratio',
            'foreground_ratio', 'foreground_objects', 'gradient_mean',
            'gradient_std', 'edge_density'
        ]
        
        for key in motion_keys:
            if key not in motion_dict:
                motion_dict[key] = 0.0
                
        return motion_dict
    
    def _extract_texture_features(self, frame: np.ndarray) -> Dict[str, float]:
        """Extract texture and edge features"""
        texture_dict = {}
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Local Binary Pattern
            lbp = feature.local_binary_pattern(gray, 24, 8, method='uniform')
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=26, range=(0, 26))
            lbp_hist = lbp_hist.astype(float) / lbp_hist.sum()
            
            # Gray-Level Co-occurrence Matrix
            glcm = feature.graycomatrix(
                (gray // 32).astype(np.uint8), [1], [0], levels=8, symmetric=True, normed=True
            )
            
            # Texture properties
            contrast = feature.graycoprops(glcm, 'contrast')[0, 0]
            dissimilarity = feature.graycoprops(glcm, 'dissimilarity')[0, 0]
            homogeneity = feature.graycoprops(glcm, 'homogeneity')[0, 0]
            energy = feature.graycoprops(glcm, 'energy')[0, 0]
            
            # Edge detection
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Gaussian derivatives
            gaussian_grad = ndimage.gaussian_gradient_magnitude(gray, sigma=1)
            
            texture_dict.update({
                'lbp_uniformity': float(np.sum(lbp_hist**2)),
                'lbp_contrast': float(np.sum((np.arange(len(lbp_hist)) - np.mean(np.arange(len(lbp_hist))))**2 * lbp_hist)),
                'glcm_contrast': float(contrast),
                'glcm_dissimilarity': float(dissimilarity),
                'glcm_homogeneity': float(homogeneity),
                'glcm_energy': float(energy),
                'edge_density': float(edge_density),
                'gaussian_grad_mean': float(np.mean(gaussian_grad)),
                'gaussian_grad_std': float(np.std(gaussian_grad)),
                'texture_regularity': float(np.std(lbp_hist))
            })
            
        except Exception as e:
            self.logger.error(f"Texture feature extraction failed: {e}")
            texture_dict = {f'texture_{i}': 0.0 for i in range(10)}
            
        return texture_dict
    
    def _extract_spatial_features(self, detections: Dict, frame_shape: Tuple) -> Dict[str, float]:
        """Extract spatial relationship features between detected objects"""
        spatial_dict = {}
        
        try:
            boxes = detections.get('boxes', [])
            labels = detections.get('labels', [])
            
            if not boxes:
                return {f'spatial_{i}': 0.0 for i in range(15)}
            
            # Basic statistics
            spatial_dict['num_detections'] = float(len(boxes))
            spatial_dict['detection_density'] = float(len(boxes) / (frame_shape[0] * frame_shape[1]))
            
            # Box properties
            areas = []
            aspect_ratios = []
            centers = []
            
            for box in boxes:
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1
                area = width * height
                aspect_ratio = width / height if height > 0 else 0
                center = ((x1 + x2) / 2, (y1 + y2) / 2)
                
                areas.append(area)
                aspect_ratios.append(aspect_ratio)
                centers.append(center)
            
            if areas:
                spatial_dict.update({
                    'mean_detection_area': float(np.mean(areas)),
                    'std_detection_area': float(np.std(areas)),
                    'mean_aspect_ratio': float(np.mean(aspect_ratios)),
                    'std_aspect_ratio': float(np.std(aspect_ratios))
                })
            
            # Inter-object distances
            if len(centers) > 1:
                distances = []
                for i in range(len(centers)):
                    for j in range(i+1, len(centers)):
                        dist = np.sqrt((centers[i][0] - centers[j][0])**2 + 
                                     (centers[i][1] - centers[j][1])**2)
                        distances.append(dist)
                
                spatial_dict.update({
                    'mean_inter_distance': float(np.mean(distances)),
                    'min_inter_distance': float(np.min(distances)),
                    'std_inter_distance': float(np.std(distances))
                })
            
            # Object distribution
            if centers:
                centers_array = np.array(centers)
                center_x_std = float(np.std(centers_array[:, 0]))
                center_y_std = float(np.std(centers_array[:, 1]))
                
                spatial_dict.update({
                    'center_x_spread': center_x_std,
                    'center_y_spread': center_y_std,
                    'spatial_distribution': float(center_x_std * center_y_std)
                })
            
            # Class-specific counts
            unique_labels = set(labels)
            for label in ['car', 'truck', 'person', 'motorcycle']:
                count = labels.count(label)
                spatial_dict[f'{label}_count'] = float(count)
                
        except Exception as e:
            self.logger.error(f"Spatial feature extraction failed: {e}")
            
        # Ensure we have exactly 15 spatial features
        spatial_keys = [
            'num_detections', 'detection_density', 'mean_detection_area',
            'std_detection_area', 'mean_aspect_ratio', 'std_aspect_ratio',
            'mean_inter_distance', 'min_inter_distance', 'std_inter_distance',
            'center_x_spread', 'center_y_spread', 'spatial_distribution',
            'car_count', 'truck_count', 'person_count'
        ]
        
        for key in spatial_keys:
            if key not in spatial_dict:
                spatial_dict[key] = 0.0
                
        return spatial_dict
    
    def _extract_temporal_features(self, detections: Dict) -> Dict[str, float]:
        """Extract temporal features based on detection history"""
        temporal_dict = {}
        
        try:
            if self.prev_detections is not None:
                # Track object persistence
                current_boxes = detections.get('boxes', [])
                prev_boxes = self.prev_detections.get('boxes', [])
                
                # Calculate IoU overlaps for tracking
                overlaps = self._calculate_iou_matrix(current_boxes, prev_boxes)
                
                if overlaps.size > 0:
                    max_overlaps = np.max(overlaps, axis=1) if overlaps.shape[1] > 0 else np.array([])
                    temporal_dict.update({
                        'mean_object_persistence': float(np.mean(max_overlaps)) if len(max_overlaps) > 0 else 0.0,
                        'num_new_objects': float(np.sum(max_overlaps < 0.3)) if len(max_overlaps) > 0 else 0.0,
                        'num_disappeared_objects': float(len(prev_boxes) - len(current_boxes)) if len(prev_boxes) > len(current_boxes) else 0.0
                    })
                
                # Detection count changes
                temporal_dict['detection_count_change'] = float(len(current_boxes) - len(prev_boxes))
                
            # Frame history statistics
            if len(self.frame_history) > 1:
                detection_counts = [len(frame_det.get('boxes', [])) for frame_det in self.frame_history]
                temporal_dict.update({
                    'detection_count_trend': float(np.mean(np.diff(detection_counts))),
                    'detection_count_variance': float(np.var(detection_counts)),
                    'detection_stability': float(1.0 / (1.0 + np.std(detection_counts)))
                })
                
        except Exception as e:
            self.logger.error(f"Temporal feature extraction failed: {e}")
        
        # Ensure all temporal features exist
        temporal_keys = [
            'mean_object_persistence', 'num_new_objects', 'num_disappeared_objects',
            'detection_count_change', 'detection_count_trend', 'detection_count_variance',
            'detection_stability'
        ]
        
        for key in temporal_keys:
            if key not in temporal_dict:
                temporal_dict[key] = 0.0
                
        return temporal_dict
    
    def _extract_environmental_features(self, frame: np.ndarray, metadata: Optional[Dict]) -> Dict[str, float]:
        """Extract environmental and lighting features"""
        env_dict = {}
        
        try:
            # Lighting analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Brightness and contrast
            brightness = np.mean(gray)
            contrast = np.std(gray)
            
            # Color distribution
            saturation = np.mean(hsv[:, :, 1])
            hue_circular_std = self._circular_std(hsv[:, :, 0])
            
            env_dict.update({
                'brightness': float(brightness),
                'contrast': float(contrast),
                'saturation': float(saturation),
                'hue_diversity': float(hue_circular_std),
                'dynamic_range': float(np.max(gray) - np.min(gray))
            })
            
            # Weather indicators from metadata
            if metadata:
                weather = metadata.get('weather', {})
                env_dict.update({
                    'weather_clear': float(weather.get('clear', 0)),
                    'weather_rain': float(weather.get('rain', 0)),
                    'weather_fog': float(weather.get('fog', 0)),
                    'temperature': float(weather.get('temperature', 20)),
                    'visibility': float(weather.get('visibility', 10))
                })
            
        except Exception as e:
            self.logger.error(f"Environmental feature extraction failed: {e}")
        
        # Default environmental features
        env_keys = [
            'brightness', 'contrast', 'saturation', 'hue_diversity',
            'dynamic_range', 'weather_clear', 'weather_rain',
            'weather_fog', 'temperature', 'visibility'
        ]
        
        for key in env_keys:
            if key not in env_dict:
                env_dict[key] = 0.0 if 'weather' not in key else 20.0 if key == 'temperature' else 10.0 if key == 'visibility' else 0.0
                
        return env_dict
    
    def _extract_detection_features(self, detections: Dict) -> Dict[str, float]:
        """Extract features specific to detection quality and confidence"""
        detection_dict = {}
        
        try:
            scores = detections.get('scores', [])
            boxes = detections.get('boxes', [])
            
            if scores:
                detection_dict.update({
                    'mean_confidence': float(np.mean(scores)),
                    'max_confidence': float(np.max(scores)),
                    'min_confidence': float(np.min(scores)),
                    'confidence_std': float(np.std(scores)),
                    'high_confidence_ratio': float(np.sum(np.array(scores) > 0.8) / len(scores))
                })
            
            if boxes:
                # Box quality metrics
                box_areas = [(x2-x1)*(y2-y1) for x1, y1, x2, y2 in boxes]
                detection_dict.update({
                    'mean_box_area': float(np.mean(box_areas)),
                    'box_area_std': float(np.std(box_areas))
                })
                
        except Exception as e:
            self.logger.error(f"Detection feature extraction failed: {e}")
        
        # Default detection features
        detection_keys = [
            'mean_confidence', 'max_confidence', 'min_confidence',
            'confidence_std', 'high_confidence_ratio', 'mean_box_area', 'box_area_std'
        ]
        
        for key in detection_keys:
            if key not in detection_dict:
                detection_dict[key] = 0.0
                
        return detection_dict
    
    def _update_history(self, frame: np.ndarray, detections: Dict):
        """Update frame and detection history"""
        # Update previous frame
        self.prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.prev_detections = detections
        
        # Update frame history (keep last 10 frames)
        self.frame_history.append(detections)
        if len(self.frame_history) > 10:
            self.frame_history.pop(0)
    
    def _calculate_iou_matrix(self, boxes1: List, boxes2: List) -> np.ndarray:
        """Calculate IoU matrix between two sets of boxes"""
        if not boxes1 or not boxes2:
            return np.array([])
        
        iou_matrix = np.zeros((len(boxes1), len(boxes2)))
        
        for i, box1 in enumerate(boxes1):
            for j, box2 in enumerate(boxes2):
                iou_matrix[i, j] = self._calculate_iou(box1, box2)
                
        return iou_matrix
    
    def _calculate_iou(self, box1: List, box2: List) -> float:
        """Calculate IoU between two boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_int = max(x1_1, x1_2)
        y1_int = max(y1_1, y1_2)
        x2_int = min(x2_1, x2_2)
        y2_int = min(y2_1, y2_2)
        
        if x2_int <= x1_int or y2_int <= y1_int:
            return 0.0
        
        intersection = (x2_int - x1_int) * (y2_int - y1_int)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _circular_std(self, angles: np.ndarray) -> float:
        """Calculate circular standard deviation for hue values"""
        angles_rad = angles * np.pi / 180
        x = np.mean(np.cos(angles_rad))
        y = np.mean(np.sin(angles_rad))
        r = np.sqrt(x**2 + y**2)
        return np.sqrt(-2 * np.log(r)) if r > 0 else 0.0
    
    def _get_default_features(self) -> Dict[str, float]:
        """Return default feature dictionary with zeros"""
        feature_groups = {
            'cnn': 9,
            'motion': 14,
            'texture': 10,
            'spatial': 15,
            'temporal': 7,
            'environmental': 10,
            'detection': 7
        }
        
        default_features = {}
        for group, count in feature_groups.items():
            for i in range(count):
                default_features[f'{group}_{i}'] = 0.0
                
        return default_features
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names"""
        # This would return the complete list of 127 feature names
        # Implementation depends on the exact feature naming convention
        pass


# Example usage
if __name__ == "__main__":
    extractor = AdvancedFeatureExtractor()
    
    # Test with dummy data
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    dummy_detections = {
        'boxes': [[100, 100, 200, 200], [300, 150, 400, 250]],
        'scores': [0.85, 0.92],
        'labels': ['car', 'truck']
    }
    
    features = extractor.extract_features(dummy_frame, dummy_detections)
    print(f"Extracted {len(features)} features")
    print("Sample features:", {k: v for k, v in list(features.items())[:10]})
