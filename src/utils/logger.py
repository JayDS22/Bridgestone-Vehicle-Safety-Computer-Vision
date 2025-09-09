"""
Logging utilities for vehicle safety system
Centralized logging configuration with structured logging
"""

import logging
import logging.handlers
import sys
import os
from datetime import datetime
from pathlib import Path
import json
import structlog
from typing import Dict, Optional, Any

def setup_logger(name: str, 
                level: str = "INFO",
                log_file: Optional[str] = None,
                log_dir: str = "logs",
                max_bytes: int = 10485760,  # 10MB
                backup_count: int = 5) -> logging.Logger:
    """
    Setup structured logger with file and console handlers
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Log file name (optional)
        log_dir: Log directory
        max_bytes: Maximum log file size
        backup_count: Number of backup files
        
    Returns:
        Configured logger
    """
    
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="ISO"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Get logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = VehicleSafetyFormatter()
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_path = log_path / log_file
    else:
        file_path = log_path / f"{name}.log"
    
    file_handler = logging.handlers.RotatingFileHandler(
        file_path,
        maxBytes=max_bytes,
        backupCount=backup_count
    )
    file_formatter = StructuredJSONFormatter()
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Error file handler
    error_file_path = log_path / f"{name}_errors.log"
    error_handler = logging.handlers.RotatingFileHandler(
        error_file_path,
        maxBytes=max_bytes,
        backupCount=backup_count
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_formatter)
    logger.addHandler(error_handler)
    
    logger.info(f"Logger {name} initialized", extra={
        "log_level": level,
        "log_file": str(file_path),
        "max_bytes": max_bytes
    })
    
    return logger


class VehicleSafetyFormatter(logging.Formatter):
    """
    Custom formatter for console output with colors and structured format
    """
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }
    
    def format(self, record):
        # Add color to level name
        level_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        colored_level = f"{level_color}{record.levelname:8}{self.COLORS['RESET']}"
        
        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')
        
        # Format message
        message = record.getMessage()
        
        # Add extra context if available
        extra_info = ""
        if hasattr(record, 'inference_time'):
            extra_info += f" [inference: {record.inference_time:.1f}ms]"
        if hasattr(record, 'detection_count'):
            extra_info += f" [detections: {record.detection_count}]"
        if hasattr(record, 'risk_score'):
            extra_info += f" [risk: {record.risk_score:.3f}]"
        
        # Construct final message
        formatted = f"{timestamp} | {colored_level} | {record.name:20} | {message}{extra_info}"
        
        # Add exception info if present
        if record.exc_info:
            formatted += f"\n{self.formatException(record.exc_info)}"
        
        return formatted


class StructuredJSONFormatter(logging.Formatter):
    """
    JSON formatter for structured logging to files
    """
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'process_id': record.process,
            'thread_id': record.thread
        }
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
                          'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                          'thread', 'threadName', 'processName', 'process', 'getMessage']:
                log_entry[key] = value
        
        # Add exception info
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry, default=str)


class PerformanceLogger:
    """
    Specialized logger for performance metrics
    """
    
    def __init__(self, logger_name: str = "performance"):
        self.logger = logging.getLogger(logger_name)
        self.metrics_file = Path("logs") / "performance_metrics.jsonl"
        
        # Ensure metrics file exists
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
    
    def log_inference_performance(self, 
                                 inference_time: float,
                                 detection_count: int,
                                 risk_score: float,
                                 model_version: str = "unknown",
                                 additional_metrics: Optional[Dict] = None):
        """
        Log inference performance metrics
        
        Args:
            inference_time: Inference time in milliseconds
            detection_count: Number of objects detected
            risk_score: Calculated risk score
            model_version: Model version identifier
            additional_metrics: Additional metrics to log
        """
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'type': 'inference_performance',
            'inference_time_ms': inference_time,
            'detection_count': detection_count,
            'risk_score': risk_score,
            'model_version': model_version,
            'meets_sla': inference_time < 150,  # 150ms SLA
        }
        
        if additional_metrics:
            metrics.update(additional_metrics)
        
        # Log to main logger
        self.logger.info(
            "Inference completed",
            extra=metrics
        )
        
        # Append to metrics file
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(metrics, default=str) + '\n')
    
    def log_training_metrics(self,
                           model_name: str,
                           training_duration: float,
                           metrics: Dict[str, float],
                           dataset_size: int):
        """
        Log training performance metrics
        
        Args:
            model_name: Name of the model
            training_duration: Training duration in seconds
            metrics: Training metrics (accuracy, loss, etc.)
            dataset_size: Size of training dataset
        """
        
        training_log = {
            'timestamp': datetime.now().isoformat(),
            'type': 'training_performance',
            'model_name': model_name,
            'training_duration_seconds': training_duration,
            'dataset_size': dataset_size,
            **metrics
        }
        
        self.logger.info(
            f"Training completed for {model_name}",
            extra=training_log
        )
        
        # Append to metrics file
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(training_log, default=str) + '\n')
    
    def log_system_metrics(self,
                          cpu_percent: float,
                          memory_percent: float,
                          disk_usage_percent: float,
                          gpu_percent: Optional[float] = None):
        """
        Log system resource metrics
        
        Args:
            cpu_percent: CPU usage percentage
            memory_percent: Memory usage percentage
            disk_usage_percent: Disk usage percentage
            gpu_percent: GPU usage percentage (optional)
        """
        
        system_metrics = {
            'timestamp': datetime.now().isoformat(),
            'type': 'system_performance',
            'cpu_percent': cpu_percent,
            'memory_percent': memory_percent,
            'disk_usage_percent': disk_usage_percent
        }
        
        if gpu_percent is not None:
            system_metrics['gpu_percent'] = gpu_percent
        
        self.logger.debug(
            "System metrics",
            extra=system_metrics
        )
        
        # Append to metrics file
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(system_metrics, default=str) + '\n')


class AuditLogger:
    """
    Audit logger for tracking critical system events
    """
    
    def __init__(self, logger_name: str = "audit"):
        self.logger = logging.getLogger(logger_name)
        self.audit_file = Path("logs") / "audit.log"
        
        # Setup audit file handler
        self.audit_file.parent.mkdir(parents=True, exist_ok=True)
        audit_handler = logging.handlers.RotatingFileHandler(
            self.audit_file,
            maxBytes=50*1024*1024,  # 50MB
            backupCount=10
        )
        audit_formatter = StructuredJSONFormatter()
        audit_handler.setFormatter(audit_formatter)
        
        self.logger.addHandler(audit_handler)
        self.logger.setLevel(logging.INFO)
    
    def log_model_prediction(self,
                           request_id: str,
                           input_hash: str,
                           prediction: Dict[str, Any],
                           user_id: Optional[str] = None):
        """
        Log model prediction for audit trail
        
        Args:
            request_id: Unique request identifier
            input_hash: Hash of input data
            prediction: Model prediction results
            user_id: User identifier (optional)
        """
        
        audit_entry = {
            'event_type': 'model_prediction',
            'request_id': request_id,
            'input_hash': input_hash,
            'prediction_summary': {
                'risk_score': prediction.get('risk_score'),
                'detection_count': len(prediction.get('vehicles', {}).get('boxes', [])),
                'processing_time': prediction.get('processing_time')
            },
            'user_id': user_id,
            'timestamp': datetime.now().isoformat()
        }
        
        self.logger.info("Model prediction", extra=audit_entry)
    
    def log_model_update(self,
                        model_name: str,
                        old_version: str,
                        new_version: str,
                        user_id: str):
        """
        Log model updates
        
        Args:
            model_name: Name of the model
            old_version: Previous model version
            new_version: New model version
            user_id: User who performed the update
        """
        
        audit_entry = {
            'event_type': 'model_update',
            'model_name': model_name,
            'old_version': old_version,
            'new_version': new_version,
            'user_id': user_id,
            'timestamp': datetime.now().isoformat()
        }
        
        self.logger.warning("Model updated", extra=audit_entry)
    
    def log_system_access(self,
                         user_id: str,
                         action: str,
                         resource: str,
                         success: bool,
                         ip_address: Optional[str] = None):
        """
        Log system access events
        
        Args:
            user_id: User identifier
            action: Action performed
            resource: Resource accessed
            success: Whether access was successful
            ip_address: User's IP address
        """
        
        audit_entry = {
            'event_type': 'system_access',
            'user_id': user_id,
            'action': action,
            'resource': resource,
            'success': success,
            'ip_address': ip_address,
            'timestamp': datetime.now().isoformat()
        }
        
        log_level = logging.INFO if success else logging.WARNING
        self.logger.log(log_level, f"System access: {action}", extra=audit_entry)


# Global logger instances
performance_logger = PerformanceLogger()
audit_logger = AuditLogger()

# Example usage
if __name__ == "__main__":
    # Test logger setup
    logger = setup_logger("test_logger", level="DEBUG")
    
    # Test different log levels
    logger.debug("Debug message")
    logger.info("Info message", extra={'key': 'value'})
    logger.warning("Warning message")
    logger.error("Error message")
    
    # Test performance logging
    performance_logger.log_inference_performance(
        inference_time=125.5,
        detection_count=3,
        risk_score=0.75,
        model_version="v2.1.0"
    )
    
    # Test audit logging
    audit_logger.log_model_prediction(
        request_id="req_123",
        input_hash="abc123def",
        prediction={'risk_score': 0.75, 'vehicles': {'boxes': []}},
        user_id="user_456"
    )
    
    print("Logging system initialized and tested successfully")
