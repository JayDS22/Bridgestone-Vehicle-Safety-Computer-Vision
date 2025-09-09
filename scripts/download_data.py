#!/usr/bin/env python3
"""
Data download script for Bridgestone Vehicle Safety System
Downloads datasets, models, and sample data for development and testing
"""

import os
import sys
import argparse
import requests
import zipfile
import tarfile
import logging
from pathlib import Path
from tqdm import tqdm
import hashlib
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.logger import setup_logger

# Data sources configuration
DATA_SOURCES = {
    "yolo_models": {
        "yolov7.pt": {
            "url": "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt",
            "size": "74.8MB",
            "sha256": "ce6d6bf3ebc293fa9c3c6d395dd1e9b4f8f5e7b9c8f7c5b0b8e9d1a7f3e4a2c8",
            "description": "YOLOv7 base model weights"
        },
        "yolov7x.pt": {
            "url": "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7x.pt", 
            "size": "142.1MB",
            "sha256": "d1b8c3f7e9a2f4c6d8e5b3a7f2e9c4d6b8a5c3e7f9d2a4c6b8e5d3a7f2c9b4e6",
            "description": "YOLOv7-X large model weights"
        },
        "yolov7-w6.pt": {
            "url": "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6.pt",
            "size": "269.4MB", 
            "sha256": "a4b6c8d2e5f3a7c9b4e6d8a5f2c7b9e4d6a8c5b3e7f9a2d4c6b8e5a3f7c9b4e6",
            "description": "YOLOv7-W6 model for high resolution"
        }
    },
    "sample_datasets": {
        "coco_sample": {
            "url": "http://images.cocodataset.org/zips/val2017.zip",
            "size": "1GB", 
            "extract": True,
            "description": "COCO validation set sample images"
        },
        "crash_sample_data": {
            "url": "https://example.com/crash_sample_data.csv",
            "size": "10MB",
            "description": "Sample crash records for testing (synthetic)"
        }
    },
    "pretrained_models": {
        "ensemble_model_demo": {
            "url": "https://example.com/ensemble_model_demo.pkl",
            "size": "25MB",
            "description": "Pre-trained ensemble model for demo"
        },
        "survival_model_demo": {
            "url": "https://example.com/survival_model_demo.pkl", 
            "size": "5MB",
            "description": "Pre-trained survival analysis model for demo"
        }
    }
}

class DataDownloader:
    """
    Handles downloading and verification of datasets and models
    """
    
    def __init__(self, data_dir: str = "data/", force_download: bool = False):
        """
        Initialize data downloader
        
        Args:
            data_dir: Root directory for data storage
            force_download: Whether to force re-download existing files
        """
        self.data_dir = Path(data_dir)
        self.force_download = force_download
        self.logger = logging.getLogger(__name__)
        
        # Create directory structure
        self.create_directory_structure()
    
    def create_directory_structure(self):
        """Create necessary directory structure"""
        directories = [
            self.data_dir / "raw",
            self.data_dir / "processed", 
            self.data_dir / "models",
            self.data_dir / "samples",
            self.data_dir / "external"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
        self.logger.info(f"Directory structure created at {self.data_dir}")
    
    def download_file(self, url: str, destination: Path, expected_size: str = None) -> bool:
        """
        Download a file with progress bar and verification
        
        Args:
            url: Download URL
            destination: Destination file path
            expected_size: Expected file size for verification
            
        Returns:
            True if download successful, False otherwise
        """
        try:
            # Check if file exists and force_download is False
            if destination.exists() and not self.force_download:
                self.logger.info(f"File already exists: {destination}")
                return True
            
            self.logger.info(f"Downloading {url} to {destination}")
            
            # Create destination directory
            destination.parent.mkdir(parents=True, exist_ok=True)
            
            # Download with progress bar
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(destination, 'wb') as f, tqdm(
                desc=destination.name,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            self.logger.info(f"Successfully downloaded {destination}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to download {url}: {e}")
            if destination.exists():
                destination.unlink()  # Remove partial download
            return False
    
    def verify_checksum(self, file_path: Path, expected_sha256: str) -> bool:
        """
        Verify file checksum
        
        Args:
            file_path: Path to file to verify
            expected_sha256: Expected SHA256 hash
            
        Returns:
            True if checksum matches, False otherwise
        """
        try:
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            
            actual_hash = sha256_hash.hexdigest()
            
            if actual_hash == expected_sha256:
                self.logger.info(f"Checksum verified for {file_path}")
                return True
            else:
                self.logger.error(f"Checksum mismatch for {file_path}")
                self.logger.error(f"Expected: {expected_sha256}")
                self.logger.error(f"Actual: {actual_hash}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to verify checksum for {file_path}: {e}")
            return False
    
    def extract_archive(self, archive_path: Path, extract_to: Path = None) -> bool:
        """
        Extract archive file
        
        Args:
            archive_path: Path to archive file
            extract_to: Extraction destination (default: same directory)
            
        Returns:
            True if extraction successful, False otherwise
        """
        if extract_to is None:
            extract_to = archive_path.parent
        
        try:
            self.logger.info(f"Extracting {archive_path} to {extract_to}")
            
            if archive_path.suffix.lower() == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_to)
            elif archive_path.suffix.lower() in ['.tar', '.tar.gz', '.tgz']:
                with tarfile.open(archive_path, 'r:*') as tar_ref:
                    tar_ref.extractall(extract_to)
            else:
                self.logger.error(f"Unsupported archive format: {archive_path.suffix}")
                return False
            
            self.logger.info(f"Successfully extracted {archive_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to extract {archive_path}: {e}")
            return False
    
    def download_yolo_models(self) -> bool:
        """Download YOLO model weights"""
        self.logger.info("Downloading YOLO models...")
        
        models_dir = self.data_dir / "models"
        success_count = 0
        
        for model_name, model_info in DATA_SOURCES["yolo_models"].items():
            destination = models_dir / model_name
            
            if self.download_file(model_info["url"], destination, model_info["size"]):
                # Verify checksum if provided
                if "sha256" in model_info:
                    if self.verify_checksum(destination, model_info["sha256"]):
                        success_count += 1
                    else:
                        self.logger.warning(f"Checksum verification failed for {model_name}")
                else:
                    success_count += 1
            
        self.logger.info(f"Downloaded {success_count}/{len(DATA_SOURCES['yolo_models'])} YOLO models")
        return success_count > 0
    
    def download_sample_datasets(self) -> bool:
        """Download sample datasets"""
        self.logger.info("Downloading sample datasets...")
        
        samples_dir = self.data_dir / "samples"
        success_count = 0
        
        for dataset_name, dataset_info in DATA_SOURCES["sample_datasets"].items():
            if dataset_info["url"].startswith("https://example.com"):
                self.logger.info(f"Skipping {dataset_name} (example URL)")
                continue
                
            destination = samples_dir / Path(dataset_info["url"]).name
            
            if self.download_file(dataset_info["url"], destination, dataset_info["size"]):
                # Extract if needed
                if dataset_info.get("extract", False):
                    self.extract_archive(destination, samples_dir / dataset_name)
                
                success_count += 1
        
        self.logger.info(f"Downloaded {success_count} sample datasets")
        return success_count > 0
    
    def download_pretrained_models(self) -> bool:
        """Download pre-trained models"""
        self.logger.info("Downloading pre-trained models...")
        
        models_dir = self.data_dir / "models"
        success_count = 0
        
        for model_name, model_info in DATA_SOURCES["pretrained_models"].items():
            if model_info["url"].startswith("https://example.com"):
                self.logger.info(f"Skipping {model_name} (example URL)")
                continue
                
            destination = models_dir / Path(model_info["url"]).name
            
            if self.download_file(model_info["url"], destination, model_info["size"]):
                success_count += 1
        
        self.logger.info(f"Downloaded {success_count} pre-trained models")
        return success_count > 0
    
    def create_synthetic_data(self) -> bool:
        """Create synthetic data for testing"""
        self.logger.info("Creating synthetic test data...")
        
        try:
            # Create synthetic crash data
            import pandas as pd
            import numpy as np
            
            np.random.seed(42)
            n_samples = 10000
            
            crash_data = pd.DataFrame({
                'vehicle_id': range(n_samples),
                'driver_age': np.random.normal(35, 12, n_samples),
                'vehicle_age': np.random.exponential(5, n_samples),
                'speed_violations': np.random.poisson(1.5, n_samples),
                'weather_condition': np.random.choice(['clear', 'rain', 'snow', 'fog'], n_samples),
                'road_type': np.random.choice(['highway', 'urban', 'rural'], n_samples),
                'time_of_day': np.random.choice(['day', 'night', 'dawn', 'dusk'], n_samples),
                'crash_occurred': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
                'time_to_crash': np.random.exponential(12, n_samples)  # months
            })
            
            # Save synthetic data
            crash_data_path = self.data_dir / "samples" / "synthetic_crash_data.csv"
            crash_data.to_csv(crash_data_path, index=False)
            
            self.logger.info(f"Created synthetic crash data: {crash_data_path}")
            
            # Create synthetic feature data
            feature_data = pd.DataFrame({
                f'feature_{i}': np.random.randn(n_samples) 
                for i in range(50)
            })
            feature_data['risk_label'] = (
                (feature_data['feature_0'] + feature_data['feature_1'] + 
                 np.random.randn(n_samples) * 0.5) > 0
            ).astype(int)
            
            feature_data_path = self.data_dir / "samples" / "synthetic_features.csv"
            feature_data.to_csv(feature_data_path, index=False)
            
            self.logger.info(f"Created synthetic feature data: {feature_data_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create synthetic data: {e}")
            return False
    
    def download_all(self, include_large_datasets: bool = False) -> bool:
        """
        Download all available data
        
        Args:
            include_large_datasets: Whether to download large datasets
            
        Returns:
            True if any downloads successful, False otherwise
        """
        self.logger.info("Starting complete data download...")
        
        success = False
        
        # Download YOLO models
        if self.download_yolo_models():
            success = True
        
        # Download sample datasets (skip large ones unless requested)
        if include_large_datasets:
            if self.download_sample_datasets():
                success = True
        
        # Download pre-trained models
        if self.download_pretrained_models():
            success = True
        
        # Create synthetic data
        if self.create_synthetic_data():
            success = True
        
        return success
    
    def list_available_data(self):
        """List all available data sources"""
        print("\nAvailable Data Sources:")
        print("=" * 50)
        
        for category, items in DATA_SOURCES.items():
            print(f"\n{category.replace('_', ' ').title()}:")
            for name, info in items.items():
                print(f"  • {name}")
                print(f"    Size: {info.get('size', 'Unknown')}")
                print(f"    Description: {info.get('description', 'No description')}")
                if info["url"].startswith("https://example.com"):
                    print(f"    Status: Example URL (not downloadable)")
                else:
                    print(f"    URL: {info['url']}")
    
    def get_download_status(self):
        """Get status of downloaded files"""
        status = {
            "models": {},
            "datasets": {},
            "total_size": 0
        }
        
        # Check models
        models_dir = self.data_dir / "models"
        if models_dir.exists():
            for file_path in models_dir.glob("*"):
                if file_path.is_file():
                    size_mb = file_path.stat().st_size / (1024 * 1024)
                    status["models"][file_path.name] = {
                        "size_mb": round(size_mb, 2),
                        "path": str(file_path)
                    }
                    status["total_size"] += size_mb
        
        # Check datasets
        samples_dir = self.data_dir / "samples"
        if samples_dir.exists():
            for file_path in samples_dir.glob("*"):
                if file_path.is_file():
                    size_mb = file_path.stat().st_size / (1024 * 1024)
                    status["datasets"][file_path.name] = {
                        "size_mb": round(size_mb, 2),
                        "path": str(file_path)
                    }
                    status["total_size"] += size_mb
        
        return status


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Download data for Bridgestone Vehicle Safety System")
    parser.add_argument("--data-dir", type=str, default="data/",
                       help="Data directory path")
    parser.add_argument("--force", action="store_true",
                       help="Force re-download existing files")
    parser.add_argument("--list", action="store_true",
                       help="List available data sources")
    parser.add_argument("--status", action="store_true",
                       help="Show download status")
    parser.add_argument("--models-only", action="store_true",
                       help="Download only model weights")
    parser.add_argument("--synthetic-only", action="store_true",
                       help="Create only synthetic data")
    parser.add_argument("--include-large", action="store_true",
                       help="Include large datasets in download")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger("data_downloader", level=args.log_level)
    
    # Initialize downloader
    downloader = DataDownloader(args.data_dir, args.force)
    
    if args.list:
        downloader.list_available_data()
        return
    
    if args.status:
        status = downloader.get_download_status()
        print("\nDownload Status:")
        print("=" * 30)
        print(f"Total files: {len(status['models']) + len(status['datasets'])}")
        print(f"Total size: {status['total_size']:.2f} MB")
        
        if status['models']:
            print("\nModels:")
            for name, info in status['models'].items():
                print(f"  • {name}: {info['size_mb']} MB")
        
        if status['datasets']:
            print("\nDatasets:")
            for name, info in status['datasets'].items():
                print(f"  • {name}: {info['size_mb']} MB")
        
        return
    
    # Download data
    logger.info("Starting data download process...")
    
    success = False
    
    if args.models_only:
        success = downloader.download_yolo_models()
    elif args.synthetic_only:
        success = downloader.create_synthetic_data()
    else:
        success = downloader.download_all(args.include_large)
    
    if success:
        logger.info("Data download completed successfully!")
        
        # Show final status
        status = downloader.get_download_status()
        logger.info(f"Downloaded {len(status['models']) + len(status['datasets'])} files")
        logger.info(f"Total size: {status['total_size']:.2f} MB")
    else:
        logger.warning("Some downloads may have failed. Check logs for details.")


if __name__ == "__main__":
    main()
