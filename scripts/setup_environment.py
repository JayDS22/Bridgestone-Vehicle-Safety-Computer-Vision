#!/usr/bin/env python3
"""
Environment setup script for Bridgestone Vehicle Safety System
Sets up development environment, dependencies, and configurations
"""

import os
import sys
import subprocess
import platform
import argparse
import logging
from pathlib import Path
import json
import yaml

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def setup_logger():
    """Setup basic logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

class EnvironmentSetup:
    """
    Handles complete environment setup for the vehicle safety system
    """
    
    def __init__(self, project_root: str = None):
        """
        Initialize environment setup
        
        Args:
            project_root: Root directory of the project
        """
        self.project_root = Path(project_root) if project_root else Path(__file__).parent.parent
        self.logger = setup_logger()
        self.system_info = self.get_system_info()
        
    def get_system_info(self) -> dict:
        """Get system information"""
        info = {
            'platform': platform.platform(),
            'system': platform.system(),
            'machine': platform.machine(),
            'python_version': platform.python_version(),
            'python_executable': sys.executable
        }
        
        # Check for GPU availability
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            info['gpu_available'] = result.returncode == 0
            if info['gpu_available']:
                # Parse GPU info
                gpu_info = result.stdout
                info['gpu_info'] = gpu_info.split('\n')[8:12]  # Approximate GPU info lines
        except:
            info['gpu_available'] = False
            info['gpu_info'] = None
        
        return info
    
    def check_python_version(self) -> bool:
        """Check if Python version is compatible"""
        min_version = (3, 8)
        current_version = sys.version_info[:2]
        
        if current_version >= min_version:
            self.logger.info(f"✅ Python version {'.'.join(map(str, current_version))} is compatible")
            return True
        else:
            self.logger.error(f"❌ Python version {'.'.join(map(str, current_version))} is too old. Minimum required: {'.'.join(map(str, min_version))}")
            return False
    
    def create_directory_structure(self):
        """Create necessary directory structure"""
        directories = [
            'data/raw',
            'data/processed', 
            'data/models',
            'data/samples',
            'logs',
            'results',
            'notebooks',
            'tests',
            'config',
            'scripts',
            'deployment'
        ]
        
        self.logger.info("Creating directory structure...")
        
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            
            # Create __init__.py for Python packages
            if directory.startswith(('src/', 'tests/')):
                init_file = dir_path / '__init__.py'
                if not init_file.exists():
                    init_file.touch()
        
        self.logger.info("✅ Directory structure created")
    
    def setup_virtual_environment(self, venv_name: str = "venv") -> bool:
        """Setup Python virtual environment"""
        venv_path = self.project_root / venv_name
        
        if venv_path.exists():
            self.logger.info(f"Virtual environment already exists at {venv_path}")
            return True
        
        try:
            self.logger.info(f"Creating virtual environment at {venv_path}")
            subprocess.run([sys.executable, '-m', 'venv', str(venv_path)], check=True)
            
            # Get activation script path
            if self.system_info['system'] == 'Windows':
                activate_script = venv_path / 'Scripts' / 'activate.bat'
                pip_path = venv_path / 'Scripts' / 'pip.exe'
            else:
                activate_script = venv_path / 'bin' / 'activate'
                pip_path = venv_path / 'bin' / 'pip'
            
            self.logger.info(f"✅ Virtual environment created")
            self.logger.info(f"To activate: source {activate_script}")
            
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"❌ Failed to create virtual environment: {e}")
            return False
    
    def install_dependencies(self, requirements_file: str = "requirements.txt") -> bool:
        """Install Python dependencies"""
        requirements_path = self.project_root / requirements_file
        
        if not requirements_path.exists():
            self.logger.error(f"❌ Requirements file not found: {requirements_path}")
            return False
        
        try:
            self.logger.info("Installing Python dependencies...")
            
            # Upgrade pip first
            subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'], check=True)
            
            # Install requirements
            subprocess.run([
                sys.executable, '-m', 'pip', 'install', '-r', str(requirements_path)
            ], check=True)
            
            self.logger.info("✅ Dependencies installed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"❌ Failed to install dependencies: {e}")
            return False
    
    def setup_gpu_support(self) -> bool:
        """Setup GPU support if available"""
        if not self.system_info['gpu_available']:
            self.logger.info("No GPU detected, skipping GPU setup")
            return True
        
        try:
            self.logger.info("Setting up GPU support...")
            
            # Install PyTorch with CUDA support
            if self.system_info['system'] == 'Linux':
                cuda_command = [
                    sys.executable, '-m', 'pip', 'install', 
                    'torch', 'torchvision', 'torchaudio', 
                    '--index-url', 'https://download.pytorch.org/whl/cu118'
                ]
                subprocess.run(cuda_command, check=True)
                
                self.logger.info("✅ GPU support configured")
                return True
            else:
                self.logger.info("GPU setup not automated for this platform")
                return True
                
        except subprocess.CalledProcessError as e:
            self.logger.error(f"❌ Failed to setup GPU support: {e}")
            return False
    
    def create_default_configs(self):
        """Create default configuration files"""
        self.logger.info("Creating default configuration files...")
        
        # Create .env file
        env_file = self.project_root / '.env'
        if not env_file.exists():
            env_content = """# Bridgestone Vehicle Safety Environment Variables
ENVIRONMENT=development
LOG_LEVEL=INFO
MODEL_PATH=data/models/
CUDA_VISIBLE_DEVICES=0

# AWS Configuration (optional)
# AWS_REGION=us-east-1
# AWS_PROFILE=default

# Database Configuration (optional)
# DATABASE_URL=postgresql://user:password@localhost:5432/vehicle_safety

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
"""
            env_file.write_text(env_content)
            self.logger.info("✅ Created .env file")
        
        # Create .gitignore if it doesn't exist
        gitignore_file = self.project_root / '.gitignore'
        if not gitignore_file.exists():
            gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Data and Models
data/raw/*
data/processed/*
data/models/*.pt
data/models/*.pkl
!data/models/.gitkeep

# Logs
logs/
*.log

# Results
results/
*.png
*.jpg
*.jpeg

# Environment
.env
.env.local

# OS
.DS_Store
Thumbs.db

# Jupyter Notebooks
.ipynb_checkpoints/

# MLflow
mlruns/
mlflow.db

# Temporary files
*.tmp
*.temp
"""
            gitignore_file.write_text(gitignore_content)
            self.logger.info("✅ Created .gitignore file")
    
    def setup_pre_commit_hooks(self) -> bool:
        """Setup pre-commit hooks for code quality"""
        try:
            self.logger.info("Setting up pre-commit hooks...")
            
            # Install pre-commit
            subprocess.run([sys.executable, '-m', 'pip', 'install', 'pre-commit'], check=True)
            
            # Create .pre-commit-config.yaml
            precommit_config = self.project_root / '.pre-commit-config.yaml'
            if not precommit_config.exists():
                config_content = """repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-merge-conflict
  
  - repo: https://github.com/psf/black
    rev: 22.10.0
    hooks:
      - id: black
        language_version: python3
  
  - repo: https://github.com/pycqa/flake8
    rev: 5.0.4
    hooks:
      - id: flake8
        args: [--max-line-length=100, --ignore=E203,W503]
  
  - repo: https://github.com/pycqa/isort
    rev: 5.10.1
    hooks:
      - id: isort
        args: ["--profile", "black"]
"""
                precommit_config.write_text(config_content)
            
            # Install pre-commit hooks
            subprocess.run(['pre-commit', 'install'], cwd=self.project_root, check=True)
            
            self.logger.info("✅ Pre-commit hooks configured")
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"❌ Failed to setup pre-commit hooks: {e}")
            return False
    
    def verify_installation(self) -> bool:
        """Verify that everything is installed correctly"""
        self.logger.info("Verifying installation...")
        
        verification_results = {}
        
        # Test imports
        test_imports = [
            'numpy',
            'pandas', 
            'torch',
            'cv2',
            'sklearn',
            'fastapi',
            'uvicorn'
        ]
        
        for module in test_imports:
            try:
                __import__(module)
                verification_results[module] = True
                self.logger.info(f"✅ {module} imported successfully")
            except ImportError as e:
                verification_results[module] = False
                self.logger.error(f"❌ Failed to import {module}: {e}")
        
        # Test CUDA availability
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            verification_results['cuda'] = cuda_available
            if cuda_available:
                self.logger.info(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            else:
                self.logger.info("ℹ️ CUDA not available (CPU-only mode)")
        except:
            verification_results['cuda'] = False
        
        # Check project structure
        required_dirs = ['src', 'data', 'config', 'tests']
        for directory in required_dirs:
            dir_path = self.project_root / directory
            verification_results[f'dir_{directory}'] = dir_path.exists()
            if dir_path.exists():
                self.logger.info(f"✅ Directory exists: {directory}")
            else:
                self.logger.error(f"❌ Missing directory: {directory}")
        
        # Overall success
        success_rate = sum(verification_results.values()) / len(verification_results)
        
        if success_rate > 0.8:
            self.logger.info(f"✅ Installation verification passed ({success_rate:.1%} success rate)")
            return True
        else:
            self.logger.error(f"❌ Installation verification failed ({success_rate:.1%} success rate)")
            return False
    
    def create_jupyter_config(self):
        """Create Jupyter notebook configuration"""
        try:
            self.logger.info("Setting up Jupyter configuration...")
            
            # Install Jupyter extensions
            subprocess.run([
                sys.executable, '-m', 'pip', 'install', 
                'jupyter', 'jupyterlab', 'ipykernel'
            ], check=True)
            
            # Create kernel for this project
            subprocess.run([
                sys.executable, '-m', 'ipykernel', 'install', '--user', 
                '--name', 'bridgestone-vehicle-safety',
                '--display-name', 'Bridgestone Vehicle Safety'
            ], check=True)
            
            self.logger.info("✅ Jupyter configuration complete")
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"❌ Failed to setup Jupyter: {e}")
    
    def print_summary(self):
        """Print setup summary and next steps"""
        print("\n" + "="*60)
        print("🚗 BRIDGESTONE VEHICLE SAFETY SYSTEM SETUP COMPLETE 🚗")
        print("="*60)
        
        print(f"\nProject Root: {self.project_root}")
        print(f"Python Version: {self.system_info['python_version']}")
        print(f"Platform: {self.system_info['platform']}")
        print(f"GPU Available: {'Yes' if self.system_info['gpu_available'] else 'No'}")
        
        print("\n📁 Project Structure:")
        print("├── src/                 # Source code")
        print("├── data/                # Data storage")
        print("├── config/              # Configuration files")
        print("├── tests/               # Test files")
        print("├── logs/                # Log files")
        print("├── results/             # Results and outputs")
        print("└── notebooks/           # Jupyter notebooks")
        
        print("\n🚀 Next Steps:")
        print("1. Activate virtual environment:")
        if self.system_info['system'] == 'Windows':
            print("   .\\venv\\Scripts\\activate")
        else:
            print("   source venv/bin/activate")
        
        print("\n2. Download sample data:")
        print("   python scripts/download_data.py --synthetic-only")
        
        print("\n3. Run tests:")
        print("   pytest tests/ -v")
        
        print("\n4. Start the API server:")
        print("   python src/api/inference_api.py")
        
        print("\n5. Open Jupyter Lab:")
        print("   jupyter lab")
        
        print("\n📖 Documentation:")
        print("   • README.md - Project overview and setup")
        print("   • config/ - Configuration examples")
        print("   • notebooks/ - Example notebooks")
        
        print("\n" + "="*60)


def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description="Setup Bridgestone Vehicle Safety development environment")
    parser.add_argument("--project-root", type=str, default=None,
                       help="Project root directory")
    parser.add_argument("--skip-venv", action="store_true",
                       help="Skip virtual environment creation")
    parser.add_argument("--skip-gpu", action="store_true", 
                       help="Skip GPU setup")
    parser.add_argument("--skip-precommit", action="store_true",
                       help="Skip pre-commit hooks setup")
    parser.add_argument("--skip-jupyter", action="store_true",
                       help="Skip Jupyter setup")
    parser.add_argument("--verify-only", action="store_true",
                       help="Only run verification")
    
    args = parser.parse_args()
    
    # Initialize setup
    setup = EnvironmentSetup(args.project_root)
    
    if args.verify_only:
        setup.verify_installation()
        return
    
    # Check Python version
    if not setup.check_python_version():
        return
    
    # Create directory structure
    setup.create_directory_structure()
    
    # Setup virtual environment
    if not args.skip_venv:
        setup.setup_virtual_environment()
    
    # Install dependencies
    setup.install_dependencies()
    
    # Setup GPU support
    if not args.skip_gpu:
        setup.setup_gpu_support()
    
    # Create default configs
    setup.create_default_configs()
    
    # Setup pre-commit hooks
    if not args.skip_precommit:
        setup.setup_pre_commit_hooks()
    
    # Setup Jupyter
    if not args.skip_jupyter:
        setup.create_jupyter_config()
    
    # Verify installation
    setup.verify_installation()
    
    # Print summary
    setup.print_summary()


if __name__ == "__main__":
    main()
