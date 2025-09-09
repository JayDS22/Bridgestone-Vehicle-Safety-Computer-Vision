"""
Performance Monitoring and Metrics Collection
Real-time monitoring for production ML pipeline
"""

import time
import numpy as np
import psutil
import logging
from typing import Dict, List, Optional
from collections import deque, defaultdict
from datetime import datetime, timedelta
import threading
import json

class PerformanceMonitor:
    """
    Comprehensive performance monitoring for the vehicle safety system
    
    Tracks:
    - Inference times
    - Throughput
    - Memory usage
    - CPU usage
    - Error rates
    - Detection accuracy
    """
    
    def __init__(self, max_history: int = 10000):
        """
        Initialize performance monitor
        
        Args:
            max_history: Maximum number of records to keep in memory
        """
        self.max_history = max_history
        
        # Performance metrics storage
        self.inference_times = deque(maxlen=max_history)
        self.throughput_data = deque(maxlen=max_history)
        self.memory_usage = deque(maxlen=max_history)
        self.cpu_usage = deque(maxlen=max_history)
        self.error_counts = defaultdict(int)
        self.detection_stats = deque(maxlen=max_history)
        
        # System monitoring
        self.start_time = time.time()
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        
        # Thread-safe locks
        self.lock = threading.Lock()
        
        # System monitoring thread
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._system_monitor, daemon=True)
        self.monitor_thread.start()
        
        self.logger = logging.getLogger(__name__)
    
    def log_inference(self, 
                     inference_time: float, 
                     detection_count: int,
                     success: bool = True,
                     error_type: Optional[str] = None):
        """
        Log inference performance metrics
        
        Args:
            inference_time: Inference time in milliseconds
            detection_count: Number of objects detected
            success: Whether inference was successful
            error_type: Type of error if failed
        """
        with self.lock:
            timestamp = time.time()
            
            # Update counters
            self.total_requests += 1
            if success:
                self.successful_requests += 1
                
                # Log successful inference
                self.inference_times.append({
                    'time': timestamp,
                    'inference_time_ms': inference_time,
                    'detection_count': detection_count
                })
                
                self.detection_stats.append({
                    'time': timestamp,
                    'count': detection_count
                })
                
            else:
                self.failed_requests += 1
                if error_type:
                    self.error_counts[error_type] += 1
            
            # Calculate throughput (requests per second)
            current_throughput = 1000 / inference_time if inference_time > 0 else 0
            self.throughput_data.append({
                'time': timestamp,
                'throughput': current_throughput
            })
    
    def _system_monitor(self):
        """Background thread for system resource monitoring"""
        while self.monitoring_active:
            try:
                with self.lock:
                    timestamp = time.time()
                    
                    # Memory usage
                    memory_info = psutil.virtual_memory()
                    self.memory_usage.append({
                        'time': timestamp,
                        'percent': memory_info.percent,
                        'available_gb': memory_info.available / (1024**3),
                        'used_gb': memory_info.used / (1024**3)
                    })
                    
                    # CPU usage
                    cpu_percent = psutil.cpu_percent(interval=None)
                    self.cpu_usage.append({
                        'time': timestamp,
                        'percent': cpu_percent
                    })
                
                time.sleep(5)  # Monitor every 5 seconds
                
            except Exception as e:
                self.logger.error(f"System monitoring error: {e}")
                time.sleep(10)
    
    def get_metrics(self) -> Dict:
        """
        Get comprehensive performance metrics
        
        Returns:
            Dictionary containing all performance metrics
        """
        with self.lock:
            current_time = time.time()
            uptime = current_time - self.start_time
            
            # Calculate inference statistics
            inference_stats = self._calculate_inference_stats()
            throughput_stats = self._calculate_throughput_stats()
            system_stats = self._calculate_system_stats()
            detection_stats = self._calculate_detection_stats()
            
            return {
                'system': {
                    'uptime_seconds': uptime,
                    'uptime_hours': uptime / 3600,
                    'timestamp': datetime.now().isoformat()
                },
                'requests': {
                    'total': self.total_requests,
                    'successful': self.successful_requests,
                    'failed': self.failed_requests,
                    'success_rate': self.successful_requests / self.total_requests if self.total_requests > 0 else 0,
                    'error_rate': self.failed_requests / self.total_requests if self.total_requests > 0 else 0
                },
                'inference': inference_stats,
                'throughput': throughput_stats,
                'system_resources': system_stats,
                'detections': detection_stats,
                'errors': dict(self.error_counts)
            }
    
    def _calculate_inference_stats(self) -> Dict:
        """Calculate inference time statistics"""
        if not self.inference_times:
            return {}
        
        times = [record['inference_time_ms'] for record in self.inference_times]
        
        return {
            'mean_ms': float(np.mean(times)),
            'median_ms': float(np.median(times)),
            'p95_ms': float(np.percentile(times, 95)),
            'p99_ms': float(np.percentile(times, 99)),
            'min_ms': float(np.min(times)),
            'max_ms': float(np.max(times)),
            'std_ms': float(np.std(times)),
            'count': len(times),
            'target_met_150ms': sum(1 for t in times if t < 150) / len(times) if times else 0
        }
    
    def _calculate_throughput_stats(self) -> Dict:
        """Calculate throughput statistics"""
        if not self.throughput_data:
            return {}
        
        # Recent throughput (last 1 minute)
        recent_time = time.time() - 60
        recent_throughput = [
            record['throughput'] for record in self.throughput_data 
            if record['time'] > recent_time
        ]
        
        all_throughput = [record['throughput'] for record in self.throughput_data]
        
        stats = {
            'current_rps': float(np.mean(recent_throughput)) if recent_throughput else 0,
            'average_rps': float(np.mean(all_throughput)) if all_throughput else 0,
            'max_rps': float(np.max(all_throughput)) if all_throughput else 0,
            'target_met_1000rps': self._check_throughput_target(recent_throughput)
        }
        
        return stats
    
    def _calculate_system_stats(self) -> Dict:
        """Calculate system resource statistics"""
        current_stats = {}
        
        # Current memory usage
        if self.memory_usage:
            latest_memory = self.memory_usage[-1]
            current_stats['memory'] = {
                'percent': latest_memory['percent'],
                'available_gb': latest_memory['available_gb'],
                'used_gb': latest_memory['used_gb']
            }
        
        # Recent CPU usage (last 5 minutes)
        recent_time = time.time() - 300
        recent_cpu = [
            record['percent'] for record in self.cpu_usage 
            if record['time'] > recent_time
        ]
        
        if recent_cpu:
            current_stats['cpu'] = {
                'current_percent': float(np.mean(recent_cpu[-5:])) if recent_cpu else 0,
                'average_percent': float(np.mean(recent_cpu)),
                'max_percent': float(np.max(recent_cpu))
            }
        
        return current_stats
    
    def _calculate_detection_stats(self) -> Dict:
        """Calculate detection statistics"""
        if not self.detection_stats:
            return {}
        
        counts = [record['count'] for record in self.detection_stats]
        
        return {
            'average_detections_per_frame': float(np.mean(counts)),
            'max_detections_per_frame': int(np.max(counts)),
            'frames_with_detections': sum(1 for c in counts if c > 0) / len(counts) if counts else 0,
            'total_detections': sum(counts)
        }
    
    def _check_throughput_target(self, throughput_data: List[float]) -> bool:
        """Check if throughput target of 1000 RPS is met"""
        if not throughput_data:
            return False
        return np.mean(throughput_data) >= 1000
    
    def get_performance_report(self, time_window_hours: int = 24) -> Dict:
        """
        Generate detailed performance report
        
        Args:
            time_window_hours: Time window for report in hours
            
        Returns:
            Detailed performance report
        """
        with self.lock:
            cutoff_time = time.time() - (time_window_hours * 3600)
            
            # Filter data by time window
            recent_inferences = [
                record for record in self.inference_times 
                if record['time'] > cutoff_time
            ]
            
            recent_throughput = [
                record for record in self.throughput_data 
                if record['time'] > cutoff_time
            ]
            
            recent_detections = [
                record for record in self.detection_stats 
                if record['time'] > cutoff_time
            ]
            
            # Generate report
            report = {
                'report_period': {
                    'hours': time_window_hours,
                    'start_time': datetime.fromtimestamp(cutoff_time).isoformat(),
                    'end_time': datetime.now().isoformat()
                },
                'summary': {
                    'total_requests': len(recent_inferences),
                    'average_inference_time_ms': float(np.mean([r['inference_time_ms'] for r in recent_inferences])) if recent_inferences else 0,
                    'average_throughput_rps': float(np.mean([r['throughput'] for r in recent_throughput])) if recent_throughput else 0,
                    'sla_compliance': {
                        'inference_time_150ms': self._calculate_sla_compliance(recent_inferences, 150),
                        'throughput_1000rps': np.mean([r['throughput'] for r in recent_throughput]) >= 1000 if recent_throughput else False
                    }
                },
                'trends': self._calculate_trends(recent_inferences, recent_throughput),
                'anomalies': self._detect_anomalies(recent_inferences)
            }
            
            return report
    
    def _calculate_sla_compliance(self, inference_data: List[Dict], threshold_ms: float) -> float:
        """Calculate SLA compliance percentage"""
        if not inference_data:
            return 0.0
        
        compliant_requests = sum(
            1 for record in inference_data 
            if record['inference_time_ms'] <= threshold_ms
        )
        
        return compliant_requests / len(inference_data)
    
    def _calculate_trends(self, inference_data: List[Dict], throughput_data: List[Dict]) -> Dict:
        """Calculate performance trends"""
        if len(inference_data) < 2:
            return {}
        
        # Calculate hourly averages
        hourly_performance = self._group_by_hour(inference_data)
        
        trends = {}
        if len(hourly_performance) >= 2:
            recent_avg = np.mean([hour['avg_inference_time'] for hour in hourly_performance[-2:]])
            older_avg = np.mean([hour['avg_inference_time'] for hour in hourly_performance[:-2]]) if len(hourly_performance) > 2 else recent_avg
            
            trends['inference_time_trend'] = 'improving' if recent_avg < older_avg else 'degrading' if recent_avg > older_avg else 'stable'
        
        return trends
    
    def _group_by_hour(self, data: List[Dict]) -> List[Dict]:
        """Group data by hour for trend analysis"""
        hourly_data = defaultdict(list)
        
        for record in data:
            hour = datetime.fromtimestamp(record['time']).replace(minute=0, second=0, microsecond=0)
            hourly_data[hour].append(record)
        
        hourly_averages = []
        for hour, records in sorted(hourly_data.items()):
            avg_inference_time = np.mean([r['inference_time_ms'] for r in records])
            hourly_averages.append({
                'hour': hour.isoformat(),
                'avg_inference_time': avg_inference_time,
                'request_count': len(records)
            })
        
        return hourly_averages
    
    def _detect_anomalies(self, inference_data: List[Dict]) -> List[Dict]:
        """Detect performance anomalies"""
        if not inference_data:
            return []
        
        times = [record['inference_time_ms'] for record in inference_data]
        mean_time = np.mean(times)
        std_time = np.std(times)
        
        anomalies = []
        for record in inference_data:
            if abs(record['inference_time_ms'] - mean_time) > 3 * std_time:
                anomalies.append({
                    'timestamp': datetime.fromtimestamp(record['time']).isoformat(),
                    'inference_time_ms': record['inference_time_ms'],
                    'type': 'slow_inference' if record['inference_time_ms'] > mean_time else 'fast_inference',
                    'deviation_factor': abs(record['inference_time_ms'] - mean_time) / std_time
                })
        
        return anomalies[-10:]  # Return last 10 anomalies
    
    def export_metrics(self, filepath: str):
        """Export metrics to JSON file"""
        metrics = self.get_metrics()
        
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        self.logger.info(f"Metrics exported to {filepath}")
    
    def reset_metrics(self):
        """Reset all metrics (use with caution)"""
        with self.lock:
            self.inference_times.clear()
            self.throughput_data.clear()
            self.detection_stats.clear()
            self.error_counts.clear()
            self.total_requests = 0
            self.successful_requests = 0
            self.failed_requests = 0
            self.start_time = time.time()
        
        self.logger.info("Metrics reset")
    
    def stop_monitoring(self):
        """Stop background monitoring thread"""
        self.monitoring_active = False
        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)


class AlertManager:
    """
    Alert management for performance monitoring
    """
    
    def __init__(self, monitor: PerformanceMonitor):
        self.monitor = monitor
        self.alert_thresholds = {
            'inference_time_ms': 200,
            'memory_percent': 85,
            'cpu_percent': 80,
            'error_rate': 0.05,
            'throughput_rps': 500
        }
        self.active_alerts = set()
        self.logger = logging.getLogger(__name__)
    
    def check_alerts(self) -> List[Dict]:
        """Check for alert conditions"""
        alerts = []
        metrics = self.monitor.get_metrics()
        
        # Check inference time
        if 'inference' in metrics and 'mean_ms' in metrics['inference']:
            if metrics['inference']['mean_ms'] > self.alert_thresholds['inference_time_ms']:
                alert = {
                    'type': 'high_inference_time',
                    'severity': 'warning',
                    'message': f"Average inference time {metrics['inference']['mean_ms']:.1f}ms exceeds threshold {self.alert_thresholds['inference_time_ms']}ms",
                    'timestamp': datetime.now().isoformat()
                }
                alerts.append(alert)
        
        # Check memory usage
        if 'system_resources' in metrics and 'memory' in metrics['system_resources']:
            memory_percent = metrics['system_resources']['memory']['percent']
            if memory_percent > self.alert_thresholds['memory_percent']:
                alert = {
                    'type': 'high_memory_usage',
                    'severity': 'critical' if memory_percent > 95 else 'warning',
                    'message': f"Memory usage {memory_percent:.1f}% exceeds threshold {self.alert_thresholds['memory_percent']}%",
                    'timestamp': datetime.now().isoformat()
                }
                alerts.append(alert)
        
        # Check error rate
        if 'requests' in metrics:
            error_rate = metrics['requests']['error_rate']
            if error_rate > self.alert_thresholds['error_rate']:
                alert = {
                    'type': 'high_error_rate',
                    'severity': 'critical',
                    'message': f"Error rate {error_rate:.2%} exceeds threshold {self.alert_thresholds['error_rate']:.2%}",
                    'timestamp': datetime.now().isoformat()
                }
                alerts.append(alert)
        
        return alerts
    
    def set_threshold(self, metric: str, value: float):
        """Update alert threshold"""
        if metric in self.alert_thresholds:
            self.alert_thresholds[metric] = value
            self.logger.info(f"Updated alert threshold for {metric} to {value}")


# Example usage
if __name__ == "__main__":
    # Initialize monitor
    monitor = PerformanceMonitor()
    
    # Simulate some inference calls
    import random
    for i in range(100):
        inference_time = random.uniform(50, 200)
        detection_count = random.randint(0, 10)
        success = random.random() > 0.05  # 95% success rate
        
        monitor.log_inference(inference_time, detection_count, success)
        time.sleep(0.01)  # Small delay
    
    # Get metrics
    metrics = monitor.get_metrics()
    print("Performance Metrics:")
    print(json.dumps(metrics, indent=2, default=str))
    
    # Generate report
    report = monitor.get_performance_report(time_window_hours=1)
    print("\nPerformance Report:")
    print(json.dumps(report, indent=2, default=str))
    
    # Check alerts
    alert_manager = AlertManager(monitor)
    alerts = alert_manager.check_alerts()
    print(f"\nActive Alerts: {len(alerts)}")
    for alert in alerts:
        print(f"- {alert['type']}: {alert['message']}")
    
    # Cleanup
    monitor.stop_monitoring()
