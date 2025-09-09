"""
Data loading utilities for vehicle safety dataset
Handles video, image, and CSV data loading
"""

import os
import cv2
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Generator
from pathlib import Path
import logging
import json
from datetime import datetime
import h5py
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

class VehicleDataLoader:
    """
    Comprehensive data loader for vehicle safety datasets
    
    Supports:
    - Video files (MP4, AVI, MOV)
    - Image datasets (JPG, PNG, TIFF)
    - CSV/Parquet crash records
    - Real-time camera streams
    - Batch processing
    """
    
    def __init__(self, 
                 data_root: str = "data/",
                 cache_size: int = 1000,
                 num_workers: int = 4):
        """
        Initialize data loader
        
        Args:
            data_root: Root directory for data
            cache_size: Number of items to cache in memory
            num_workers: Number of worker threads for parallel loading
        """
        self.data_root = Path(data_root)
        self.cache_size = cache_size
        self.num_workers = num_workers
        
        # Data paths
        self.raw_path = self.data_root / "raw"
        self.processed_path = self.data_root / "processed"
        self.models_path = self.data_root / "models"
        
        # Create directories if they don't exist
        for path in [self.raw_path, self.processed_path, self.models_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Cache for frequently accessed data
        self._cache = {}
        self._cache_lock = threading.Lock()
        
        # Supported file formats
        self.video_formats = {'.mp4', '.avi', '.mov', '.mkv', '.flv'}
        self.image_formats = {'.jpg', '.jpeg', '.png', '.tiff', '.bmp'}
        self.data_formats = {'.csv', '.parquet', '.h5', '.hdf5'}
        
        self.logger = logging.getLogger(__name__)
        
    def load_video_dataset(self, 
                          video_dir: str,
                          metadata_file: Optional[str] = None,
                          frame_skip: int = 1,
                          max_videos: Optional[int] = None) -> Generator[Dict, None, None]:
        """
        Load video dataset with optional metadata
        
        Args:
            video_dir: Directory containing video files
            metadata_file: Optional CSV file with video metadata
            frame_skip: Skip every N frames
            max_videos: Maximum number of videos to process
            
        Yields:
            Dictionary containing video data and metadata
        """
        video_path = Path(video_dir)
        if not video_path.exists():
            raise FileNotFoundError(f"Video directory not found: {video_dir}")
        
        # Load metadata if provided
        metadata = {}
        if metadata_file and os.path.exists(metadata_file):
            meta_df = pd.read_csv(metadata_file)
            metadata = meta_df.set_index('video_filename').to_dict('index')
        
        # Find video files
        video_files = []
        for ext in self.video_formats:
            video_files.extend(list(video_path.glob(f"*{ext}")))
        
        if max_videos:
            video_files = video_files[:max_videos]
        
        self.logger.info(f"Found {len(video_files)} video files")
        
        for video_file in video_files:
            try:
                yield self._load_single_video(
                    video_file, 
                    metadata.get(video_file.name, {}),
                    frame_skip
                )
            except Exception as e:
                self.logger.error(f"Error loading video {video_file}: {e}")
                continue
    
    def _load_single_video(self, 
                          video_path: Path, 
                          metadata: Dict,
                          frame_skip: int) -> Dict:
        """Load a single video file"""
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        frames = []
        frame_timestamps = []
        frame_idx = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % frame_skip == 0:
                    frames.append(frame)
                    frame_timestamps.append(frame_idx / fps)
                
                frame_idx += 1
                
        finally:
            cap.release()
        
        return {
            'video_path': str(video_path),
            'frames': frames,
            'timestamps': frame_timestamps,
            'properties': {
                'fps': fps,
                'total_frames': frame_count,
                'width': width,
                'height': height,
                'duration_seconds': frame_count / fps
            },
            'metadata': metadata
        }
    
    def load_image_dataset(self,
                          image_dir: str,
                          annotations_file: Optional[str] = None,
                          batch_size: int = 32) -> Generator[Dict, None, None]:
        """
        Load image dataset with optional annotations
        
        Args:
            image_dir: Directory containing images
            annotations_file: Optional annotations file (COCO format or CSV)
            batch_size: Number of images per batch
            
        Yields:
            Batches of image data
        """
        image_path = Path(image_dir)
        if not image_path.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        
        # Load annotations
        annotations = {}
        if annotations_file and os.path.exists(annotations_file):
            annotations = self._load_annotations(annotations_file)
        
        # Find image files
        image_files = []
        for ext in self.image_formats:
            image_files.extend(list(image_path.glob(f"*{ext}")))
        
        self.logger.info(f"Found {len(image_files)} image files")
        
        # Process in batches
        for i in range(0, len(image_files), batch_size):
            batch_files = image_files[i:i + batch_size]
            batch_data = self._load_image_batch(batch_files, annotations)
            yield batch_data
    
    def _load_image_batch(self, image_files: List[Path], annotations: Dict) -> Dict:
        """Load a batch of images"""
        images = []
        image_info = []
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_file = {
                executor.submit(self._load_single_image, img_file): img_file 
                for img_file in image_files
            }
            
            for future in as_completed(future_to_file):
                img_file = future_to_file[future]
                try:
                    image = future.result()
                    if image is not None:
                        images.append(image)
                        image_info.append({
                            'filename': img_file.name,
                            'path': str(img_file),
                            'annotations': annotations.get(img_file.name, {})
                        })
                except Exception as e:
                    self.logger.error(f"Error loading image {img_file}: {e}")
        
        return {
            'images': images,
            'image_info': image_info,
            'batch_size': len(images)
        }
    
    def _load_single_image(self, image_path: Path) -> Optional[np.ndarray]:
        """Load a single image file"""
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                self.logger.warning(f"Could not load image: {image_path}")
                return None
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            self.logger.error(f"Error loading image {image_path}: {e}")
            return None
    
    def _load_annotations(self, annotations_file: str) -> Dict:
        """Load annotations from file"""
        file_path = Path(annotations_file)
        
        if file_path.suffix == '.json':
            # COCO format
            with open(annotations_file, 'r') as f:
                coco_data = json.load(f)
            return self._parse_coco_annotations(coco_data)
        
        elif file_path.suffix == '.csv':
            # CSV format
            df = pd.read_csv(annotations_file)
            return df.set_index('filename').to_dict('index')
        
        else:
            self.logger.warning(f"Unsupported annotation format: {file_path.suffix}")
            return {}
    
    def _parse_coco_annotations(self, coco_data: Dict) -> Dict:
        """Parse COCO format annotations"""
        annotations = {}
        
        # Create image id to filename mapping
        images = {img['id']: img['file_name'] for img in coco_data['images']}
        
        # Group annotations by image
        for ann in coco_data['annotations']:
            image_id = ann['image_id']
            filename = images.get(image_id)
            
            if filename:
                if filename not in annotations:
                    annotations[filename] = {'objects': []}
                
                annotations[filename]['objects'].append({
                    'category_id': ann['category_id'],
                    'bbox': ann['bbox'],
                    'area': ann.get('area', 0),
                    'iscrowd': ann.get('iscrowd', 0)
                })
        
        return annotations
    
    def load_crash_data(self, 
                       data_file: str,
                       sample_fraction: Optional[float] = None,
                       date_range: Optional[Tuple[str, str]] = None) -> pd.DataFrame:
        """
        Load crash records dataset
        
        Args:
            data_file: Path to crash data file (CSV, Parquet, HDF5)
            sample_fraction: Fraction of data to sample (for large datasets)
            date_range: Optional date range filter (start_date, end_date)
            
        Returns:
            DataFrame with crash records
        """
        file_path = Path(data_file)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        # Check cache first
        cache_key = f"crash_data_{file_path.stem}_{sample_fraction}_{date_range}"
        with self._cache_lock:
            if cache_key in self._cache:
                self.logger.info(f"Loading crash data from cache")
                return self._cache[cache_key].copy()
        
        # Load data based on file format
        if file_path.suffix == '.csv':
            df = pd.read_csv(data_file)
        elif file_path.suffix == '.parquet':
            df = pd.read_parquet(data_file)
        elif file_path.suffix in ['.h5', '.hdf5']:
            df = pd.read_hdf(data_file, key='crash_data')
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        self.logger.info(f"Loaded {len(df)} crash records")
        
        # Apply date range filter
        if date_range and 'crash_date' in df.columns:
            start_date, end_date = date_range
            df = df[
                (df['crash_date'] >= start_date) & 
                (df['crash_date'] <= end_date)
            ]
            self.logger.info(f"Filtered to {len(df)} records in date range")
        
        # Apply sampling
        if sample_fraction and sample_fraction < 1.0:
            df = df.sample(frac=sample_fraction, random_state=42)
            self.logger.info(f"Sampled {len(df)} records")
        
        # Cache the result
        with self._cache_lock:
            if len(self._cache) >= self.cache_size:
                # Remove oldest entry
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
            self._cache[cache_key] = df.copy()
        
        return df
    
    def load_streaming_data(self, 
                           camera_url: str,
                           buffer_size: int = 30) -> Generator[np.ndarray, None, None]:
        """
        Load real-time streaming data from camera
        
        Args:
            camera_url: Camera stream URL or device ID
            buffer_size: Buffer size for frames
            
        Yields:
            Video frames
        """
        cap = cv2.VideoCapture(camera_url)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open camera stream: {camera_url}")
        
        self.logger.info(f"Started camera stream: {camera_url}")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    self.logger.warning("Failed to read frame from stream")
                    break
                
                yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
        finally:
            cap.release()
            self.logger.info("Camera stream closed")
    
    def save_processed_data(self, 
                           data: Union[pd.DataFrame, Dict],
                           filename: str,
                           format: str = 'parquet') -> str:
        """
        Save processed data to disk
        
        Args:
            data: Data to save
            filename: Output filename
            format: Output format ('parquet', 'csv', 'hdf5')
            
        Returns:
            Path to saved file
        """
        output_path = self.processed_path / f"{filename}.{format}"
        
        if isinstance(data, pd.DataFrame):
            if format == 'parquet':
                data.to_parquet(output_path, index=False)
            elif format == 'csv':
                data.to_csv(output_path, index=False)
            elif format == 'hdf5':
                data.to_hdf(output_path, key='data', mode='w')
        
        elif isinstance(data, dict):
            if format == 'json':
                with open(output_path.with_suffix('.json'), 'w') as f:
                    json.dump(data, f, indent=2, default=str)
        
        self.logger.info(f"Saved processed data to {output_path}")
        return str(output_path)
    
    def get_dataset_info(self, dataset_path: str) -> Dict:
        """Get information about a dataset"""
        path = Path(dataset_path)
        
        if not path.exists():
            return {'error': 'Dataset path does not exist'}
        
        info = {
            'path': str(path),
            'size_bytes': sum(f.stat().st_size for f in path.rglob('*') if f.is_file()),
            'created': datetime.fromtimestamp(path.stat().st_ctime).isoformat(),
            'modified': datetime.fromtimestamp(path.stat().st_mtime).isoformat()
        }
        
        if path.is_dir():
            # Directory statistics
            file_counts = {}
            for file_path in path.rglob('*'):
                if file_path.is_file():
                    ext = file_path.suffix.lower()
                    file_counts[ext] = file_counts.get(ext, 0) + 1
            
            info.update({
                'type': 'directory',
                'total_files': sum(file_counts.values()),
                'file_types': file_counts
            })
        
        else:
            # Single file statistics
            info.update({
                'type': 'file',
                'extension': path.suffix,
                'size_mb': info['size_bytes'] / (1024 * 1024)
            })
            
            # Additional info for data files
            if path.suffix == '.csv':
                try:
                    df = pd.read_csv(path, nrows=1)
                    info['columns'] = len(df.columns)
                    info['column_names'] = list(df.columns)
                except:
                    pass
        
        return info


# Example usage
if __name__ == "__main__":
    loader = VehicleDataLoader()
    
    # Test dataset info
    info = loader.get_dataset_info("data/raw/")
    print("Dataset info:", info)
    
    # Test crash data loading
    try:
        crash_data = loader.load_crash_data("data/raw/crash_records.csv", sample_fraction=0.1)
        print(f"Loaded crash data: {crash_data.shape}")
    except FileNotFoundError:
        print("Crash data file not found - this is expected in demo")
    
    print("Data loader initialized successfully")
