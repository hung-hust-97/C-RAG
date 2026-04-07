# Language Detection Monitoring and Observability

This document describes the monitoring and observability features available in the FastText language detection system.

## Overview

The language detection system provides comprehensive monitoring capabilities including:

- **Performance Metrics**: Latency, throughput, accuracy tracking
- **Health Checks**: Component availability and status monitoring
- **Method Usage Tracking**: Distribution of detection methods used
- **Prometheus Integration**: Export metrics in Prometheus format
- **Error Tracking**: Recent errors and failure rates

## Quick Start

### Basic Usage

```python
from lightrag.language_detection import LanguageDetectionService

# Initialize service
service = LanguageDetectionService()

# Perform detections
result1 = service.detect_language("Hello world")
result2 = service.detect_language("Xin chào thế giới")

# Get performance metrics
metrics = service.get_performance_metrics()
print(f"Total detections: {metrics['total_detections']}")
print(f"Average latency: {metrics['average_processing_time_ms']}ms")
```

### Health Check

```python
# Check system health
health = service.health_check()
print(f"Status: {health['status']}")  # healthy, degraded, or unhealthy
print(f"FastText available: {health['components']['fasttext_detector']['available']}")
```

### Prometheus Metrics Export

```python
# Export metrics in Prometheus format
prometheus_metrics = service.export_prometheus_metrics()
print(prometheus_metrics)
```

## Detailed Metrics

### Performance Metrics

The `get_performance_metrics()` method returns comprehensive performance data:

```python
metrics = service.get_performance_metrics()

# Latency metrics (from detailed_metrics)
detailed = metrics['detailed_metrics']
print(f"Average latency: {detailed['avg_latency_ms']}ms")
print(f"P50 latency: {detailed['p50_latency_ms']}ms")
print(f"P95 latency: {detailed['p95_latency_ms']}ms")
print(f"P99 latency: {detailed['p99_latency_ms']}ms")

# Throughput
print(f"Detections per second: {detailed['detections_per_second']}")

# Cache performance
print(f"Cache hit rate: {detailed['cache_hit_rate_percent']}%")

# Method distribution
print(f"FastText usage: {detailed['fasttext_usage_percent']}%")
print(f"Unicode usage: {detailed['unicode_usage_percent']}%")
print(f"Hybrid usage: {detailed['hybrid_usage_percent']}%")

# Error rate
print(f"Error rate: {detailed['error_rate_percent']}%")
```

### Accuracy Tracking

You can track detection accuracy by providing ground truth labels:

```python
# Perform detection
predicted = service.detect_language("Xin chào")

# Record accuracy sample with ground truth
service.record_accuracy_sample(
    predicted=predicted,
    actual=SupportedLanguage.VIETNAMESE
)

# Get accuracy metrics
metrics = service.get_performance_metrics()
detailed = metrics['detailed_metrics']
if detailed['accuracy_percent'] is not None:
    print(f"Overall accuracy: {detailed['accuracy_percent']}%")
    print(f"Vietnamese accuracy: {detailed['vietnamese_accuracy_percent']}%")
    print(f"English accuracy: {detailed['english_accuracy_percent']}%")
```

### Recent Errors

```python
# Get recent detection errors for debugging
errors = service.get_recent_errors(limit=10)
for error in errors:
    print(f"Time: {error['timestamp']}")
    print(f"Method: {error['method_used']}")
    print(f"Error: {error['error']}")
```

## Health Checks

### Component Status

The health check provides detailed status for all components:

```python
health = service.health_check()

# Overall status
print(f"Status: {health['status']}")  # healthy, degraded, unhealthy, error
print(f"Message: {health['message']}")

# Unicode detector
unicode_status = health['components']['unicode_detector']
print(f"Unicode available: {unicode_status['available']}")
print(f"Unicode cache: {unicode_status['cache_stats']}")

# FastText detector
fasttext_status = health['components']['fasttext_detector']
print(f"FastText available: {fasttext_status['available']}")
print(f"FastText enabled: {fasttext_status['enabled']}")
if fasttext_status['available']:
    print(f"Model info: {fasttext_status['model_info']}")

# Hybrid strategy
hybrid_status = health['components']['hybrid_strategy']
print(f"Hybrid available: {hybrid_status['available']}")

# Error handler and circuit breaker
error_handler = health['components']['error_handler']
print(f"Circuit breaker open: {error_handler['circuit_breaker_open']}")
```

### Health Status Interpretation

- **healthy**: All systems operational, FastText available (if enabled)
- **degraded**: System operational but with limitations (e.g., FastText unavailable, circuit breaker open, high error rate)
- **unhealthy**: Critical components unavailable (e.g., Unicode detector failed)
- **error**: Health check itself failed

## Prometheus Integration

### Metrics Endpoint

For integration with Prometheus, expose the metrics endpoint:

```python
from fastapi import FastAPI
from lightrag.language_detection import LanguageDetectionService

app = FastAPI()
service = LanguageDetectionService()

@app.get("/metrics")
def metrics():
    """Prometheus metrics endpoint."""
    return service.export_prometheus_metrics()

@app.get("/health")
def health():
    """Health check endpoint."""
    return service.health_check()
```

### Available Prometheus Metrics

The following metrics are exported in Prometheus format:

- `lightrag_language_detection_total`: Total number of detections (counter)
- `lightrag_language_detection_errors_total`: Total number of errors (counter)
- `lightrag_language_detection_latency_seconds`: Detection latency (summary with quantiles)
- `lightrag_language_detection_throughput_per_second`: Detections per second (gauge)
- `lightrag_language_detection_cache_hit_rate`: Cache hit rate percentage (gauge)
- `lightrag_language_detection_method_total{method="..."}`: Detections by method (counter)
- `lightrag_language_detection_language_total{language="..."}`: Detections by language (counter)
- `lightrag_language_detection_accuracy`: Overall accuracy percentage (gauge, if available)
- `lightrag_language_detection_vietnamese_accuracy`: Vietnamese accuracy (gauge, if available)
- `lightrag_language_detection_english_accuracy`: English accuracy (gauge, if available)
- `lightrag_language_detection_error_rate`: Error rate percentage (gauge)

### Grafana Dashboard

Example Grafana queries:

```promql
# Detection rate
rate(lightrag_language_detection_total[5m])

# P95 latency
lightrag_language_detection_latency_seconds{quantile="0.95"}

# Error rate
lightrag_language_detection_error_rate

# Cache hit rate
lightrag_language_detection_cache_hit_rate

# Method distribution
lightrag_language_detection_method_total

# Language distribution
lightrag_language_detection_language_total
```

## Configuration

### Logging Configuration

Control logging verbosity through environment variables:

```bash
# Enable detection method logging
export LIGHTRAG_FASTTEXT_LOG_DETECTION_METHOD=true

# Enable performance metrics logging
export LIGHTRAG_FASTTEXT_LOG_PERFORMANCE_METRICS=true
```

### Metrics Retention

The metrics collector retains data for a configurable time window:

```python
from lightrag.language_detection.metrics import MetricsCollector

# Custom retention settings
collector = MetricsCollector(
    retention_window=7200,  # 2 hours
    max_samples=20000       # Maximum samples to retain
)
```

## Monitoring Best Practices

### 1. Set Up Alerts

Configure alerts for critical metrics:

- **High Error Rate**: Alert when error rate > 5%
- **High Latency**: Alert when P95 latency > 200ms
- **Low Cache Hit Rate**: Alert when cache hit rate < 70%
- **Circuit Breaker Open**: Alert when circuit breaker is open

### 2. Track Accuracy

Regularly sample detections and provide ground truth labels to track accuracy over time:

```python
# Sample 1% of detections for accuracy tracking
import random

result = service.detect_language(text)
if random.random() < 0.01:
    # Get ground truth from user or manual review
    actual_language = get_ground_truth(text)
    service.record_accuracy_sample(result, actual_language)
```

### 3. Monitor Method Distribution

Track which detection methods are being used:

```python
metrics = service.get_performance_metrics()
method_dist = metrics['method_distribution']

# Alert if FastText usage drops unexpectedly
if method_dist.get('fasttext', 0) < expected_threshold:
    alert("FastText usage below threshold")
```

### 4. Regular Health Checks

Implement periodic health checks:

```python
import schedule
import time

def check_health():
    health = service.health_check()
    if health['status'] != 'healthy':
        alert(f"Language detection unhealthy: {health['message']}")

# Check every 5 minutes
schedule.every(5).minutes.do(check_health)

while True:
    schedule.run_pending()
    time.sleep(60)
```

### 5. Reset Metrics Periodically

For long-running services, consider resetting metrics periodically to avoid memory growth:

```python
# Reset metrics daily
schedule.every().day.at("00:00").do(service.reset_metrics)
```

## Troubleshooting

### High Latency

If P95 latency is high:

1. Check cache hit rate - low hit rate increases latency
2. Verify FastText model is loaded and available
3. Check if circuit breaker is open (forcing Unicode fallback)
4. Review recent errors for patterns

### Low Accuracy

If accuracy drops:

1. Check if FastText is available (Unicode-only has lower accuracy)
2. Review method distribution - ensure hybrid strategy is being used
3. Check for changes in input text characteristics
4. Verify model file integrity

### High Error Rate

If error rate increases:

1. Check recent errors: `service.get_recent_errors()`
2. Review health check for component failures
3. Check circuit breaker status
4. Verify model file availability and permissions

## API Reference

### LanguageDetectionService

- `detect_language(text: str) -> SupportedLanguage`: Main detection method
- `get_detection_details(text: str) -> DetectionResult`: Detailed detection with metadata
- `health_check() -> Dict[str, Any]`: Comprehensive health status
- `get_performance_metrics() -> Dict[str, Any]`: Performance statistics
- `export_prometheus_metrics() -> str`: Prometheus-formatted metrics
- `get_recent_errors(limit: int = 10) -> List[Dict]`: Recent error records
- `record_accuracy_sample(predicted, actual)`: Record accuracy sample
- `reset_metrics()`: Reset all metrics and statistics

### MetricsCollector

- `record_detection(...)`: Record a detection event
- `record_accuracy(predicted, actual)`: Record accuracy sample
- `get_performance_metrics() -> PerformanceMetrics`: Get aggregated metrics
- `export_prometheus_metrics() -> str`: Export Prometheus format
- `get_recent_errors(limit: int) -> List[Dict]`: Get recent errors
- `reset_metrics()`: Reset all metrics

### HealthChecker

- `check_health() -> HealthStatus`: Perform comprehensive health check
- `check_model_availability() -> Dict`: Check FastText model status

## Examples

See `lightrag/language_detection/example_usage.py` for complete examples of:

- Basic detection with monitoring
- Health check integration
- Prometheus metrics export
- Accuracy tracking
- Error handling and debugging
