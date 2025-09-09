"""
Setup script for Bridgestone Vehicle Safety Computer Vision System
"""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# Get version
def get_version():
    version_file = os.path.join("src", "__init__.py")
    if os.path.exists(version_file):
        with open(version_file, "r") as f:
            for line in f:
                if line.startswith("__version__"):
                    return line.split("=")[1].strip().strip('"').strip("'")
    return "2.1.0"

setup(
    name="bridgestone-vehicle-safety",
    version=get_version(),
    author="Bridgestone AI/ML Engineering Team",
    author_email="ai-team@bridgestone.com",
    description="Real-time vehicle safety monitoring system using YOLOv7, computer vision, and statistical analysis",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/bridgestone/vehicle-safety-cv",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.1.0",
            "pytest-cov>=3.0.0",
            "black>=22.6.0",
            "flake8>=5.0.0",
            "mypy>=0.971",
            "pre-commit>=2.20.0",
            "jupyter>=1.0.0",
        ],
        "aws": [
            "boto3>=1.24.0",
            "awscli>=1.25.0",
        ],
        "monitoring": [
            "prometheus-client>=0.14.0",
            "grafana-api>=1.0.3",
        ],
        "gpu": [
            "torch[cuda]>=1.12.0",
            "torchvision[cuda]>=0.13.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "bridgestone-api=api.inference_api:main",
            "bridgestone-train=training.train_ensemble:main",
            "bridgestone-benchmark=scripts.benchmark:main",
            "bridgestone-process-video=scripts.process_video:main",
        ]
    },
    include_package_data=True,
    package_data={
        "": ["config/*.yaml", "config/*.yml", "*.md"],
    },
    zip_safe=False,
    keywords=[
        "computer-vision",
        "vehicle-safety",
        "yolo",
        "machine-learning",
        "survival-analysis",
        "real-time",
        "bridgestone",
        "automotive"
    ],
    project_urls={
        "Bug Reports": "https://github.com/bridgestone/vehicle-safety-cv/issues",
        "Source": "https://github.com/bridgestone/vehicle-safety-cv",
        "Documentation": "https://bridgestone-vehicle-safety.readthedocs.io/",
    },
)
