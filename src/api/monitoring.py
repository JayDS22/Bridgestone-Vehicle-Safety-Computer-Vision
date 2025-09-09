"""
API monitoring and health check utilities
Real-time monitoring for FastAPI vehicle safety service
"""

import time
import psutil
import asyncio
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging
from fastapi import HTTPException
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import json

# Prometheus metrics
REQUEST_COUNT = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('api_request_duration_seconds', 'Request duration', ['method', 'endpoint'])
INFERENCE_TIME = Histogram('inference_time_seconds', 'Model inference time')
ACTIVE_CONNECTIONS = Gauge('active_connections', 'Number of active connections')
SYSTEM_MEMORY = Gauge('system_memory_percent', 'System memory usage percentage')
SYSTEM_CPU = Gauge('system_cpu_percent', 'System CPU usage percentage')

class APIMonitor:
    """
    Comprehensive API monitoring system
    """
    
    def __init__(self):
        self.start_time = time.time()
        self.request_history = []
        self.error_history = []
        self.logger = logging.getLogger(__name__)
        
        # Health check status
        self.health_status = {
            'status': 'healthy',
            'checks': {}
        }
        
    async def record_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record API request metrics"""
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=str(status_code)).inc()
        REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)
        
        # Store in history
        self.request_history.append({
            'timestamp': datetime.now(),
            'method': method,
            'endpoint': endpoint,
            'status_code': status_code,
            'duration': duration
        })
        
        # Keep only last 1000 requests
        if len(self.request_history) > 1000:
            self.request_history.pop(0)
    
    async def record_inference(self, inference_time: float):
        """Record model inference metrics"""
        INFERENCE_TIME.observe(inference_time)
    
    async def record_error(self, error: Exception, context: Dict):
        """Record error information"""
        self.error_history.append({
            'timestamp': datetime.now(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context
        })
        
        # Keep only last 100 errors
        if len(self.error_history) > 100:
            self.error_history.pop(0)
    
    async def update_system_metrics(self):
        """Update system resource metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            SYSTEM_CPU.set(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            SYSTEM_MEMORY.set(memory.percent)
            
            self.logger.debug(f"System metrics - CPU: {cpu_percent}%, Memory: {memory.percent}%")
            
        except Exception as e:
            self.logger.error(f"Failed to update system metrics: {e}")
    
    async def health_check(self) -> Dict:
        """
        Comprehensive health check
        
        Returns:
            Health status dictionary
        """
        checks = {}
        overall_status = 'healthy'
        
        try:
            # Check system resources
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            checks['memory'] = {
                'status': 'healthy' if memory.percent < 85 else 'unhealthy',
                'usage_percent': memory.percent,
                'available_gb': memory.available / (1024**3)
            }
            
            checks['cpu'] = {
                'status': 'healthy' if cpu_percent < 80 else 'unhealthy', 
                'usage_percent': cpu_percent
            }
            
            # Check recent error rate
            recent_errors = [
                error for error in self.error_history
                if error['timestamp'] > datetime.now() - timedelta(minutes=5)
            ]
            
            error_rate = len(recent_errors) / max(len(self.request_history), 1)
            checks['error_rate'] = {
                'status': 'healthy' if error_rate < 0.05 else 'unhealthy',
                'rate': error_rate,
                'recent_errors': len(recent_errors)
            }
            
            # Check response times
            recent_requests = [
                req for req in self.request_history
                if req['timestamp'] > datetime.now() - timedelta(minutes=5)
            ]
            
            if recent_requests:
                avg_response_time = sum(req['duration'] for req in recent_requests) / len(recent_requests)
                p95_response_time = sorted([req['duration'] for req in recent_requests])[int(len(recent_requests) * 0.95)]
                
                checks['response_time'] = {
                    'status': 'healthy' if p95_response_time < 0.15 else 'unhealthy',  # 150ms SLA
                    'avg_ms': avg_response_time * 1000,
                    'p95_ms': p95_response_time * 1000
                }
            else:
                checks['response_time'] = {
                    'status': 'unknown',
                    'message': 'No recent requests'
                }
            
            # Check uptime
            uptime_seconds = time.time() - self.start_time
            checks['uptime'] = {
                'status': 'healthy',
                'seconds': uptime_seconds,
                'human_readable': str(timedelta(seconds=int(uptime_seconds)))
            }
            
            # Overall status
            if any(check.get('status') == 'unhealthy' for check in checks.values()):
                overall_status = 'unhealthy'
            elif any(check.get('status') == 'unknown' for check in checks.values()):
                overall_status = 'degraded'
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            overall_status = 'unhealthy'
            checks['health_check_error'] = str(e)
        
        return {
            'status': overall_status,
            'timestamp': datetime.now().isoformat(),
            'checks': checks
        }
    
    async def get_metrics_summary(self) -> Dict:
        """Get comprehensive metrics summary"""
        # Recent requests (last hour)
        one_hour_ago = datetime.now() - timedelta(hours=1)
        recent_requests = [
            req for req in self.request_history
            if req['timestamp'] > one_hour_ago
        ]
        
        # Calculate metrics
        total_requests = len(recent_requests)
        successful_requests = len([req for req in recent_requests if 200 <= req['status_code'] < 400])
        failed_requests = total_requests - successful_requests
        
        if total_requests > 0:
            success_rate = successful_requests / total_requests
            avg_duration = sum(req['duration'] for req in recent_requests) / total_requests
            throughput = total_requests / 3600  # requests per second over the hour
        else:
            success_rate = 0
            avg_duration = 0 
            throughput = 0
        
        return {
            'time_window': '1_hour',
            'total_requests': total_requests,
            'successful_requests': successful_requests,
            'failed_requests': failed_requests,
            'success_rate': success_rate,
            'average_duration_ms': avg_duration * 1000,
            'throughput_rps': throughput,
            'error_count': len(self.error_history),
            'uptime_seconds': time.time() - self.start_time
        }
    
    def get_prometheus_metrics(self) -> str:
        """Get Prometheus metrics in text format"""
        return generate_latest()


class PerformanceProfiler:
    """
    Performance profiling for API endpoints
    """
    
    def __init__(self):
        self.profiles = {}
        self.logger = logging.getLogger(__name__)
    
    async def profile_endpoint(self, endpoint: str, func, *args, **kwargs):
        """Profile an endpoint execution"""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        
        try:
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss
            
            execution_time = end_time - start_time
            memory_delta = end_memory - start_memory
            
            if endpoint not in self.profiles:
                self.profiles[endpoint] = []
            
            self.profiles[endpoint].append({
                'timestamp': datetime.now(),
                'execution_time': execution_time,
                'memory_delta': memory_delta
            })
            
            # Keep only last 100 profiles per endpoint
            if len(self.profiles[endpoint]) > 100:
                self.profiles[endpoint].pop(0)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Profiling failed for {endpoint}: {e}")
            raise
    
    def get_profile_summary(self, endpoint: str) -> Optional[Dict]:
        """Get performance profile summary for an endpoint"""
        if endpoint not in self.profiles:
            return None
        
        profiles = self.profiles[endpoint]
        if not profiles:
            return None
        
        execution_times = [p['execution_time'] for p in profiles]
        memory_deltas = [p['memory_delta'] for p in profiles]
        
        return {
            'endpoint': endpoint,
            'sample_count': len(profiles),
            'execution_time': {
                'mean_ms': np.mean(execution_times) * 1000,
                'median_ms': np.median(execution_times) * 1000,
                'p95_ms': np.percentile(execution_times, 95) * 1000,
                'min_ms': np.min(execution_times) * 1000,
                'max_ms': np.max(execution_times) * 1000
            },
            'memory_usage': {
                'mean_mb': np.mean(memory_deltas) / (1024*1024),
                'max_mb': np.max(memory_deltas) / (1024*1024),
                'min_mb': np.min(memory_deltas) / (1024*1024)
            }
        }


# Global instances
api_monitor = APIMonitor()
performance_profiler = PerformanceProfiler()

# Middleware for automatic monitoring
class MonitoringMiddleware:
    """FastAPI middleware for automatic monitoring"""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        start_time = time.time()
        
        # Wrap send to capture response
        status_code = 500  # Default to error
        
        async def send_wrapper(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)
        
        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as e:
            await api_monitor.record_error(e, {
                'method': scope.get('method'),
                'path': scope.get('path'),
                'timestamp': datetime.now().isoformat()
            })
            raise
        finally:
            # Record request metrics
            duration = time.time() - start_time
            await api_monitor.record_request(
                method=scope.get('method', 'UNKNOWN'),
                endpoint=scope.get('path', '/unknown'),
                status_code=status_code,
                duration=duration
            )


# Health check endpoints
async def detailed_health_check():
    """Detailed health check with all system information"""
    return await api_monitor.health_check()

async def ready_check():
    """Simple readiness check"""
    health = await api_monitor.health_check()
    if health['status'] != 'healthy':
        raise HTTPException(status_code=503, detail="Service not ready")
    return {"status": "ready"}

async def live_check():
    """Simple liveness check"""
    return {"status": "alive", "timestamp": datetime.now().isoformat()}

# Example usage
if __name__ == "__main__":
    import asyncio
    import numpy as np
    
    async def test_monitoring():
        # Test health check
        health = await api_monitor.health_check()
        print("Health check:", json.dumps(health, indent=2, default=str))
        
        # Simulate some requests
        for i in range(10):
            await api_monitor.record_request('GET', '/predict', 200, 0.1 + np.random.random() * 0.05)
            await asyncio.sleep(0.1)
        
        # Get metrics summary
        summary = await api_monitor.get_metrics_summary()
        print("Metrics summary:", json.dumps(summary, indent=2, default=str))
    
    asyncio.run(test_monitoring())
