
"""
Data preprocessing utilities for vehicle safety system
Handles image, video, and tabular data preprocessing
"""

import cv2
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif
import albumentations as A
from datetime import datetime, timedelta

class ImagePreprocessor:
    """
    Image preprocessing for vehicle detection and safety analysis
    """
    
    def __init__(self, 
                 target_size: Tuple[int, int] = (640, 640),
                 normalize: bool = True,
                 enhance_contrast: bool = True):
        """
        Initialize image preprocessor
        
        Args:
            target_size: Target image size (height, width)
            normalize: Whether to normalize pixel values
            enhance_contrast: Whether to enhance contrast
        """
        self.target_size = target_size
        self.normalize = normalize
        self.enhance_contrast = enhance_contrast
        
        # Standard preprocessing pipeline
        self.preprocessing_pipeline = A.Compose([
            A.Resize(height=target_size[0], width=target_size[1]),
            A.CLAHE(clip_limit=2.0, p=0.5) if enhance_contrast else A.NoOp(),
            A.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225]) if normalize else A.NoOp()
        ])
        
        self.logger = logging.getLogger(__name__)
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess a single image
        
        Args:
            image: Input image in RGB format
            
        Returns:
            Preprocessed image
        """
        if image is None:
            raise ValueError("Input image is None")
        
        # Apply preprocessing pipeline
        processed = self.preprocessing_pipeline(image=image)
        return processed['image']
    
    def preprocess_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Preprocess a batch of images
        
        Args:
            images: List of input images
            
        Returns:
            Batch of preprocessed images
        """
        processed_images = []
        
        for image in images:
            try:
                processed = self.preprocess_image(image)
                processed_images.append(processed)
            except Exception as e:
                self.logger.warning(f"Failed to process image: {e}")
                # Add zero image as placeholder
                processed_images.append(np.zeros((*self.target_size, 3)))
        
        return np.array(processed_images)
    
    def enhance_low_light(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance low-light images for better detection
        
        Args:
            image: Input image
            
        Returns:
            Enhanced image
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to lightness channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels and convert back to RGB
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        
        return enhanced
    
    def denoise_image(self, image: np.ndarray) -> np.ndarray:
        """
        Remove noise from image
        
        Args:
            image: Input image
            
        Returns:
            Denoised image
        """
        # Convert to grayscale for noise detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        noise_level = np.std(gray)
        
        if noise_level > 30:  # High noise
            denoised = cv2.bilateralFilter(image, 9, 75, 75)
        elif noise_level > 15:  # Medium noise
            denoised = cv2.bilateralFilter(image, 5, 50, 50)
        else:
            denoised = image  # Low noise, no filtering needed
        
        return denoised


class VideoPreprocessor:
    """
    Video preprocessing for temporal analysis
    """
    
    def __init__(self, 
                 frame_rate: Optional[int] = None,
                 stabilize: bool = True,
                 temporal_window: int = 10):
        """
        Initialize video preprocessor
        
        Args:
            frame_rate: Target frame rate (None to keep original)
            stabilize: Whether to apply video stabilization
            temporal_window: Number of frames for temporal smoothing
        """
        self.frame_rate = frame_rate
        self.stabilize = stabilize
        self.temporal_window = temporal_window
        
        # Video stabilization
        if stabilize:
            self.stabilizer = cv2.videostab.StabilizerBase()
        
        self.logger = logging.getLogger(__name__)
    
    def preprocess_video_frames(self, 
                               frames: List[np.ndarray],
                               target_fps: Optional[int] = None) -> List[np.ndarray]:
        """
        Preprocess video frames
        
        Args:
            frames: List of video frames
            target_fps: Target frame rate
            
        Returns:
            Preprocessed frames
        """
        if not frames:
            return []
        
        processed_frames = frames.copy()
        
        # Frame rate adjustment
        if target_fps and self.frame_rate:
            processed_frames = self._adjust_frame_rate(processed_frames, target_fps)
        
        # Temporal smoothing
        processed_frames = self._temporal_smooth(processed_frames)
        
        # Stabilization
        if self.stabilize:
            processed_frames = self._stabilize_frames(processed_frames)
        
        return processed_frames
    
    def _adjust_frame_rate(self, frames: List[np.ndarray], target_fps: int) -> List[np.ndarray]:
        """Adjust frame rate by sampling frames"""
        if self.frame_rate <= target_fps:
            return frames
        
        # Calculate sampling interval
        interval = self.frame_rate // target_fps
        sampled_frames = frames[::interval]
        
        self.logger.info(f"Adjusted frame rate: {len(frames)} -> {len(sampled_frames)} frames")
        return sampled_frames
    
    def _temporal_smooth(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Apply temporal smoothing to reduce noise"""
        if len(frames) < self.temporal_window:
            return frames
        
        smoothed_frames = []
        half_window = self.temporal_window // 2
        
        for i in range(len(frames)):
            start_idx = max(0, i - half_window)
            end_idx = min(len(frames), i + half_window + 1)
            window_frames = frames[start_idx:end_idx]
            
            # Average frames in window
            smoothed = np.mean(window_frames, axis=0).astype(np.uint8)
            smoothed_frames.append(smoothed)
        
        return smoothed_frames
    
    def _stabilize_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Apply video stabilization"""
        # Simple stabilization using feature tracking
        if len(frames) < 2:
            return frames
        
        stabilized_frames = [frames[0]]  # First frame unchanged
        
        # Parameters for Lucas-Kanade optical flow
        lk_params = dict(winSize=(15, 15),
                        maxLevel=2,
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        # Feature detection parameters
        feature_params = dict(maxCorners=100,
                             qualityLevel=0.3,
                             minDistance=7,
                             blockSize=7)
        
        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
        
        for i in range(1, len(frames)):
            current_gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
            
            if prev_pts is not None and len(prev_pts) > 0:
                # Calculate optical flow
                curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                    prev_gray, current_gray, prev_pts, None, **lk_params
                )
                
                # Select good points
                good_old = prev_pts[status == 1]
                good_new = curr_pts[status == 1]
                
                if len(good_old) >= 4:
                    # Calculate transformation matrix
                    transform_matrix = cv2.estimateAffinePartial2D(good_old, good_new)[0]
                    
                    if transform_matrix is not None:
                        # Apply stabilization
                        stabilized = cv2.warpAffine(frames[i], transform_matrix, 
                                                  (frames[i].shape[1], frames[i].shape[0]))
                        stabilized_frames.append(stabilized)
                    else:
                        stabilized_frames.append(frames[i])
                else:
                    stabilized_frames.append(frames[i])
                
                # Update for next iteration
                prev_gray = current_gray
                prev_pts = cv2.goodFeaturesToTrack(current_gray, mask=None, **feature_params)
            else:
                stabilized_frames.append(frames[i])
                prev_gray = current_gray
                prev_pts = cv2.goodFeaturesToTrack(current_gray, mask=None, **feature_params)
        
        return stabilized_frames


class TabularDataPreprocessor:
    """
    Preprocessing for crash records and feature data
    """
    
    def __init__(self, 
                 scaling_method: str = 'standard',
                 imputation_strategy: str = 'median',
                 feature_selection: bool = True):
        """
        Initialize tabular data preprocessor
        
        Args:
            scaling_method: 'standard', 'minmax', or 'robust'
            imputation_strategy: 'mean', 'median', 'most_frequent', or 'knn'
            feature_selection: Whether to perform feature selection
        """
        self.scaling_method = scaling_method
        self.imputation_strategy = imputation_strategy
        self.feature_selection = feature_selection
        
        # Initialize scalers and imputers
        self.scaler = self._get_scaler()
        self.imputer = self._get_imputer()
        self.feature_selector = None
        
        # Store preprocessing metadata
        self.is_fitted = False
        self.feature_names = None
        self.selected_features = None
        
        self.logger = logging.getLogger(__name__)
    
    def _get_scaler(self):
        """Get appropriate scaler based on method"""
        if self.scaling_method == 'standard':
            return StandardScaler()
        elif self.scaling_method == 'minmax':
            return MinMaxScaler()
        elif self.scaling_method == 'robust':
            return RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {self.scaling_method}")
    
    def _get_imputer(self):
        """Get appropriate imputer based on strategy"""
        if self.imputation_strategy in ['mean', 'median', 'most_frequent']:
            return SimpleImputer(strategy=self.imputation_strategy)
        elif self.imputation_strategy == 'knn':
            return KNNImputer(n_neighbors=5)
        else:
            raise ValueError(f"Unknown imputation strategy: {self.imputation_strategy}")
    
    def fit_transform(self, 
                     X: pd.DataFrame, 
                     y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fit preprocessor and transform data
        
        Args:
            X: Feature data
            y: Target variable (for feature selection)
            
        Returns:
            Preprocessed feature data
        """
        self.feature_names = X.columns.tolist()
        
        # Handle missing values
        X_imputed = pd.DataFrame(
            self.imputer.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_imputed),
            columns=X.columns,
            index=X.index
        )
        
        # Feature selection
        if self.feature_selection and y is not None:
            self.feature_selector = SelectKBest(f_classif, k='all')
            X_selected = self.feature_selector.fit_transform(X_scaled, y)
            
            # Get selected feature names
            selected_mask = self.feature_selector.get_support()
            self.selected_features = X.columns[selected_mask].tolist()
            
            X_final = pd.DataFrame(
                X_selected,
                columns=self.selected_features,
                index=X.index
            )
        else:
            X_final = X_scaled
            self.selected_features = self.feature_names
        
        self.is_fitted = True
        self.logger.info(f"Fitted preprocessor: {len(self.feature_names)} -> {len(self.selected_features)} features")
        
        return X_final
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted preprocessor
        
        Args:
            X: Feature data to transform
            
        Returns:
            Preprocessed feature data
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        # Handle missing values
        X_imputed = pd.DataFrame(
            self.imputer.transform(X),
            columns=X.columns,
            index=X.index
        )
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.transform(X_imputed),
            columns=X.columns,
            index=X.index
        )
        
        # Apply feature selection
        if self.feature_selection and self.feature_selector is not None:
            X_selected = self.feature_selector.transform(X_scaled)
            X_final = pd.DataFrame(
                X_selected,
                columns=self.selected_features,
                index=X.index
            )
        else:
            X_final = X_scaled
        
        return X_final
    
    def preprocess_crash_data(self, 
                             crash_data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess crash records data
        
        Args:
            crash_data: Raw crash data
            
        Returns:
            Preprocessed crash data
        """
        processed_data = crash_data.copy()
        
        # Handle datetime columns
        datetime_columns = ['crash_date', 'report_date', 'timestamp']
        for col in datetime_columns:
            if col in processed_data.columns:
                processed_data[col] = pd.to_datetime(processed_data[col], errors='coerce')
        
        # Extract temporal features
        if 'crash_date' in processed_data.columns:
            processed_data['crash_hour'] = processed_data['crash_date'].dt.hour
            processed_data['crash_day_of_week'] = processed_data['crash_date'].dt.dayofweek
            processed_data['crash_month'] = processed_data['crash_date'].dt.month
            processed_data['is_weekend'] = processed_data['crash_day_of_week'].isin([5, 6])
        
        # Handle categorical variables
        categorical_columns = processed_data.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if col not in datetime_columns:
                # Convert to category and then to numeric codes
                processed_data[col] = pd.Categorical(processed_data[col]).codes
        
        # Remove columns with too many missing values
        missing_threshold = 0.5
        for col in processed_data.columns:
            missing_ratio = processed_data[col].isnull().sum() / len(processed_data)
            if missing_ratio > missing_threshold:
                processed_data.drop(col, axis=1, inplace=True)
                self.logger.info(f"Dropped column {col} due to {missing_ratio:.2%} missing values")
        
        return processed_data
    
    def get_feature_importance_scores(self) -> Optional[Dict[str, float]]:
        """Get feature importance scores from feature selector"""
        if not self.feature_selection or self.feature_selector is None:
            return None
        
        scores = self.feature_selector.scores_
        selected_mask = self.feature_selector.get_support()
        
        importance_dict = {}
        for i, feature in enumerate(self.feature_names):
            if selected_mask[i]:
                importance_dict[feature] = scores[i]
        
        # Sort by importance
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))


class WeatherDataProcessor:
    """
    Process weather data for integration with vehicle safety analysis
    """
    
    def __init__(self):
        self.weather_categories = {
            'clear': ['clear', 'sunny', 'fair'],
            'cloudy': ['cloudy', 'overcast', 'partly cloudy'],
            'rain': ['rain', 'drizzle', 'shower'],
            'snow': ['snow', 'sleet', 'blizzard'],
            'fog': ['fog', 'mist', 'haze'],
            'severe': ['thunderstorm', 'tornado', 'hurricane']
        }
    
    def process_weather_conditions(self, weather_data: pd.DataFrame) -> pd.DataFrame:
        """
        Process weather conditions into standardized format
        
        Args:
            weather_data: Raw weather data
            
        Returns:
            Processed weather data
        """
        processed = weather_data.copy()
        
        # Standardize weather condition categories
        if 'weather_condition' in processed.columns:
            processed['weather_category'] = processed['weather_condition'].apply(
                self._categorize_weather
            )
        
        # Create weather severity score
        processed['weather_severity'] = processed.apply(self._calculate_weather_severity, axis=1)
        
        # Visibility impact score
        if 'visibility_km' in processed.columns:
            processed['visibility_impact'] = 1 - (processed['visibility_km'] / 10).clip(0, 1)
        
        return processed
    
    def _categorize_weather(self, condition: str) -> str:
        """Categorize weather condition"""
        if pd.isna(condition):
            return 'unknown'
        
        condition_lower = condition.lower()
        for category, keywords in self.weather_categories.items():
            if any(keyword in condition_lower for keyword in keywords):
                return category
        
        return 'other'
    
    def _calculate_weather_severity(self, row: pd.Series) -> float:
        """Calculate weather severity score (0-1)"""
        severity_scores = {
            'clear': 0.0,
            'cloudy': 0.2,
            'rain': 0.6,
            'snow': 0.8,
            'fog': 0.7,
            'severe': 1.0,
            'other': 0.3,
            'unknown': 0.3
        }
        
        base_score = severity_scores.get(row.get('weather_category', 'unknown'), 0.3)
        
        # Adjust based on wind speed
        if 'wind_speed_kmh' in row and not pd.isna(row['wind_speed_kmh']):
            wind_factor = min(row['wind_speed_kmh'] / 50, 0.3)  # Max 0.3 additional
            base_score = min(base_score + wind_factor, 1.0)
        
        # Adjust based on temperature (extreme cold/heat)
        if 'temperature_c' in row and not pd.isna(row['temperature_c']):
            temp = row['temperature_c']
            if temp < -10 or temp > 35:
                base_score = min(base_score + 0.2, 1.0)
        
        return base_score


# Example usage
if __name__ == "__main__":
    # Test image preprocessing
    img_preprocessor = ImagePreprocessor()
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    processed_img = img_preprocessor.preprocess_image(dummy_image)
    print(f"Image preprocessing: {dummy_image.shape} -> {processed_img.shape}")
    
    # Test tabular preprocessing
    tab_preprocessor = TabularDataPreprocessor()
    dummy_data = pd.DataFrame(np.random.randn(100, 10), columns=[f'feature_{i}' for i in range(10)])
    dummy_target = pd.Series(np.random.randint(0, 2, 100))
    
    processed_data = tab_preprocessor.fit_transform(dummy_data, dummy_target)
    print(f"Tabular preprocessing: {dummy_data.shape} -> {processed_data.shape}")
    
    print("Preprocessors initialized successfully")
