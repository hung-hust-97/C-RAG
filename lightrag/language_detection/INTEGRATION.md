# FastText Language Detection - Integration Guide

## Overview

This document describes how all FastText language detection components are wired together and integrated with the LightRAG system. Task 13.1 ensures that all components work seamlessly with proper dependency injection, configuration management, initialization order, and resource cleanup.

## Component Architecture

### Component Hierarchy

```
LanguageDetectionService (Main Entry Point)
├── Configuration Layer
│   ├── FastTextConfig (from environment or explicit)
│   └── Configuration Validation
├── Detection Components
│   ├── Unicode Detector (LanguageDetector - always available)
│   ├── FastText Detector (FastTextDetector - optional)
│   │   ├── FastTextModelManager
│   │   ├── InputSanitizer (security)
│   │   └── PrivacyFilter (security)
│   └── Hybrid Strategy (HybridDetectionStrategy)
├── Resilience Layer
│   ├── ResilientErrorHandler
│   ├── CircuitBreaker
│   └── Retry Logic
├── Monitoring Layer
│   ├── MetricsCollector
│   ├── HealthChecker
│   └── RateLimiter (security)
└── Caching Layer
    ├── FastText Cache (LRU with TTL)
    └── Unicode Cache (LRU)
```

## Initialization Order

The `LanguageDetectionService` initializes components in the following order to ensure proper dependency resolution:

1. **Configuration Loading** (`FastTextConfig.from_env()`)
   - Loads environment variables with `LIGHTRAG_FASTTEXT_*` prefix
   - Validates configuration parameters
   - Sets up defaults for missing values

2. **Rate Limiter** (`RateLimiter`)
   - Initializes abuse prevention system
   - Configures request limits and time windows

3. **Core Components** (`_initialize_components()`)
   - **Unicode Detector**: Always initialized first (fallback)
   - **Error Handler**: Initialized with circuit breaker and retry logic
   - **FastText Components** (if enabled):
     - FastTextModelManager (downloads/validates model)
     - FastTextDetector (loads model, sets up caching)
     - HybridDetectionStrategy (coordinates both detectors)

4. **Monitoring Components**
   - **MetricsCollector**: Tracks performance and accuracy
   - **HealthChecker**: Monitors component availability

## Configuration Flow

Configuration flows through all layers via the `FastTextConfig` object:

```python
# 1. Create configuration (from environment or explicit)
config = FastTextConfig.from_env()

# 2. Service receives configuration
service = LanguageDetectionService(config=config)

# 3. Configuration propagates to components
service.config → service.unicode_detector.cache_size
              → service.fasttext_detector.config
              → service.fasttext_detector.confidence_thresholds
              → service.hybrid_strategy.config
```

### Runtime Configuration Updates

Certain parameters can be updated at runtime without restart:

```python
# Update thresholds at runtime
service.update_configuration(
    vietnamese_threshold=0.8,
    high_confidence_threshold=0.85
)

# Changes propagate automatically to:
# - service.config
# - service.fasttext_detector.confidence_thresholds
# - service.hybrid_strategy (uses updated config)
```

## Dependency Injection

All components receive their dependencies through constructor injection:

```python
# FastTextDetector receives:
FastTextDetector(
    config=config,                    # Configuration
    error_handler=error_handler       # Resilience patterns
)

# HybridDetectionStrategy receives:
HybridDetectionStrategy(
    fasttext_detector=fasttext_detector,  # ML detector
    unicode_detector=unicode_detector,    # Fallback detector
    error_handler=error_handler           # Resilience patterns
)

# HealthChecker receives:
HealthChecker(service=service)  # Reference to main service
```

This ensures:
- **Testability**: Components can be mocked/stubbed
- **Flexibility**: Dependencies can be swapped
- **Clarity**: Dependencies are explicit

## Detection Workflow

### End-to-End Detection Flow

```
User Input
    ↓
LanguageDetectionService.detect_language(text)
    ↓
Rate Limiting Check (if client_id provided)
    ↓
Input Sanitization (security)
    ↓
Empty Text Check → Return English
    ↓
HybridDetectionStrategy.detect(text)
    ↓
├─ Short Text (<20 chars) → Unicode Detection
├─ FastText Unavailable → Unicode Detection
└─ Normal Flow:
    ├─ FastText Detection
    ├─ Confidence Analysis
    ├─ High Confidence (>0.8) → FastText Result
    ├─ Low Confidence (<0.5) → Unicode Fallback
    └─ Medium Confidence → Compare Both Methods
    ↓
Result + Metrics Recording
    ↓
Return SupportedLanguage
```

### Detailed Detection with Metadata

```python
result = service.get_detection_details(text)
# Returns DetectionResult with:
# - language: SupportedLanguage
# - confidence: float
# - method_used: str
# - processing_time_ms: float
# - cached: bool
# - fasttext_available: bool
# - unicode_fallback_used: bool
# - hybrid_strategy_applied: bool
```

## Caching Integration

### Multi-Level Caching

1. **FastText Cache** (in FastTextDetector)
   - LRU cache with configurable size
   - TTL-based expiration
   - Cache key: SHA-256 hash of preprocessed text
   - Stores: (language, confidence, timestamp)

2. **Unicode Cache** (in LanguageDetector)
   - LRU cache with configurable size
   - Cache key: SHA-256 hash of text + threshold
   - Stores: (language, threshold)

### Cache Coordination

The service coordinates caching across components:

```python
# First detection (cache miss)
result1 = service.detect_language("Test text")
# → FastText cache miss
# → Performs detection
# → Stores in FastText cache

# Second detection (cache hit)
result2 = service.detect_language("Test text")
# → FastText cache hit
# → Returns cached result immediately
```

## Error Handling and Resilience

### Graceful Degradation

The system implements multiple fallback layers:

```
FastText Detection
    ↓ (on failure)
Circuit Breaker Check
    ↓ (if open)
Unicode Detection
    ↓ (on failure)
English Default
```

### Circuit Breaker Pattern

```python
# Circuit Breaker States:
# - CLOSED: Normal operation, FastText enabled
# - OPEN: FastText disabled after 5 consecutive failures
# - HALF_OPEN: Testing recovery after 60 seconds

# Automatic recovery:
if circuit_breaker.is_open():
    # Use Unicode detection
    result = unicode_detector.detect(text)
else:
    # Try FastText
    try:
        result = fasttext_detector.detect(text)
    except Exception:
        # Circuit breaker may open
        # Falls back to Unicode
```

### Retry Logic

```python
# Exponential backoff for transient failures:
# - Attempt 1: immediate
# - Attempt 2: 1 second delay
# - Attempt 3: 2 seconds delay
# - Max delay: 60 seconds
```

## Monitoring and Observability

### Metrics Collection

The `MetricsCollector` tracks:

- **Detection Metrics**:
  - Total detections
  - Average latency
  - Method distribution (FastText/Unicode/Hybrid)
  - Language distribution (Vietnamese/English)
  
- **Performance Metrics**:
  - Cache hit rates
  - Processing times
  - Batch throughput

- **Accuracy Metrics** (when ground truth provided):
  - Precision per language
  - Recall per language
  - F1 scores

### Health Checks

```python
health = service.health_check()
# Returns:
# {
#     "status": "healthy|degraded|unhealthy",
#     "timestamp": 1234567890.0,
#     "components": {
#         "unicode_detector": {"available": true},
#         "fasttext_detector": {"available": true, "model_loaded": true},
#         "hybrid_strategy": {"available": true},
#         "error_handler": {"circuit_breaker_open": false}
#     },
#     "metrics": {...}
# }
```

### Prometheus Integration

```python
# Export metrics in Prometheus format
prometheus_metrics = service.export_prometheus_metrics()
# Returns text format compatible with Prometheus scraping:
# # HELP lightrag_language_detection_total Total language detections
# # TYPE lightrag_language_detection_total counter
# lightrag_language_detection_total{language="english"} 150
# lightrag_language_detection_total{language="vietnamese"} 75
# ...
```

## Security Integration

### Input Sanitization

```python
# All text inputs are sanitized before processing:
# 1. Length validation (max 100,000 characters)
# 2. Injection pattern detection
# 3. Control character removal
# 4. Unicode normalization
```

### Privacy Protection

```python
# Text content is NEVER logged or persisted:
# - Only metadata logged (length, hash, language)
# - No text content in error messages
# - No text content in metrics
# - Cache keys use SHA-256 hashes
```

### Rate Limiting

```python
# Prevent abuse with rate limiting:
service.detect_language(text, client_id="user_123")
# - 100 requests per minute per client (default)
# - Configurable burst allowance
# - Automatic cleanup of old entries
```

## Resource Management

### Cleanup Operations

```python
# Reset all metrics and caches
service.reset_metrics()
# - Clears all caches
# - Resets performance counters
# - Resets error statistics
# - Resets circuit breaker state

# Service continues to work after cleanup
result = service.detect_language("Test")  # Works normally
```

### Memory Management

- **Model Loading**: FastText model loaded once at startup (~200MB)
- **Cache Limits**: Configurable LRU caches prevent unbounded growth
- **Metrics Retention**: Time-windowed metrics (default 1 hour)
- **Automatic Cleanup**: Old cache entries and metrics automatically removed

## Backward Compatibility

### API Compatibility

The `LanguageDetectionService` maintains full backward compatibility with the original `LanguageDetector`:

```python
# Original API (still works)
from lightrag.language_detector import LanguageDetector
detector = LanguageDetector()
result = detector.detect("Test text")

# New API (drop-in replacement)
from lightrag.language_detection import LanguageDetectionService
service = LanguageDetectionService()
result = service.detect_language("Test text")

# Both return: SupportedLanguage.ENGLISH or SupportedLanguage.VIETNAMESE
```

### Method Compatibility

All original methods are preserved:

- `detect_language(text)` → `SupportedLanguage`
- `is_vietnamese(text)` → `bool`
- `detect_batch(texts)` → `List[SupportedLanguage]`

New methods are additive:

- `get_detection_details(text)` → `DetectionResult`
- `health_check()` → `Dict[str, Any]`
- `get_performance_metrics()` → `Dict[str, Any]`

## Integration with PromptManager

The `PromptManager` can use either detector:

```python
# Option 1: Original detector (Unicode only)
from lightrag.language_detector import LanguageDetector
manager = PromptManager(PROMPTS)
manager.detector = LanguageDetector()

# Option 2: Enhanced service (FastText + Unicode)
from lightrag.language_detection import LanguageDetectionService
manager = PromptManager(PROMPTS)
manager.detector = LanguageDetectionService()

# Both work identically from PromptManager's perspective
prompt = manager.get_prompt("key", auto_detect_text="Tiếng Việt")
```

## Usage Examples

### Basic Usage

```python
from lightrag.language_detection import LanguageDetectionService

# Initialize with defaults
service = LanguageDetectionService()

# Detect language
result = service.detect_language("This is English text")
# → SupportedLanguage.ENGLISH

result = service.detect_language("Đây là tiếng Việt")
# → SupportedLanguage.VIETNAMESE
```

### Custom Configuration

```python
from lightrag.language_detection import LanguageDetectionService, FastTextConfig

# Create custom configuration
config = FastTextConfig()
config.cache_size = 5000
config.vietnamese_threshold = 0.75
config.log_detection_method = True

# Initialize with custom config
service = LanguageDetectionService(config=config)
```

### Detailed Detection

```python
# Get detailed results
result = service.get_detection_details("Test text")

print(f"Language: {result.language.value}")
print(f"Confidence: {result.confidence:.2f}")
print(f"Method: {result.method_used}")
print(f"Time: {result.processing_time_ms:.1f}ms")
print(f"Cached: {result.cached}")
```

### Batch Processing

```python
texts = [
    "English text",
    "Tiếng Việt",
    "Another English sentence",
    "Câu tiếng Việt khác"
]

results = service.detect_batch(texts)
# → [ENGLISH, VIETNAMESE, ENGLISH, VIETNAMESE]
```

### Monitoring

```python
# Check health
health = service.health_check()
print(f"Status: {health['status']}")

# Get metrics
metrics = service.get_performance_metrics()
print(f"Total detections: {metrics['total_detections']}")
print(f"Average time: {metrics['average_processing_time_ms']:.2f}ms")

# Export for Prometheus
prometheus_output = service.export_prometheus_metrics()
```

## Testing

### Integration Tests

Run the comprehensive integration test suite:

```bash
python lightrag/language_detection/test_integration_complete.py
```

This verifies:
- ✓ Component initialization order
- ✓ Configuration flow through all layers
- ✓ Dependency injection
- ✓ End-to-end detection workflow
- ✓ Batch detection
- ✓ Caching across components
- ✓ Error handling and fallbacks
- ✓ Metrics collection
- ✓ Health checks
- ✓ Resource cleanup
- ✓ Runtime configuration updates
- ✓ Prometheus metrics export
- ✓ Rate limiting
- ✓ Backward compatibility

### Unit Tests

Individual component tests are available:

```bash
# Test service integration
python lightrag/language_detection/test_service_integration.py

# Test security features
python lightrag/language_detection/test_security.py
```

## Environment Variables

Configure the system via environment variables:

```bash
# Core functionality
export LIGHTRAG_FASTTEXT_ENABLED=true
export LIGHTRAG_FASTTEXT_CACHE_SIZE=10000
export LIGHTRAG_FASTTEXT_CACHE_TTL=3600

# Confidence thresholds
export LIGHTRAG_FASTTEXT_VIETNAMESE_THRESHOLD=0.7
export LIGHTRAG_FASTTEXT_ENGLISH_THRESHOLD=0.5
export LIGHTRAG_FASTTEXT_HIGH_CONFIDENCE_THRESHOLD=0.8
export LIGHTRAG_FASTTEXT_LOW_CONFIDENCE_THRESHOLD=0.5

# Performance
export LIGHTRAG_FASTTEXT_MAX_TEXT_LENGTH=100000
export LIGHTRAG_FASTTEXT_MIN_TEXT_LENGTH=10
export LIGHTRAG_FASTTEXT_BATCH_SIZE=32

# Model management
export LIGHTRAG_FASTTEXT_MODEL_PATH=/path/to/model
export LIGHTRAG_FASTTEXT_CACHE_DIR=~/.lightrag/models/

# Logging
export LIGHTRAG_FASTTEXT_LOG_DETECTION_METHOD=true
export LIGHTRAG_FASTTEXT_LOG_PERFORMANCE_METRICS=false

# Error handling
export LIGHTRAG_FASTTEXT_ENABLE_CIRCUIT_BREAKER=true
export LIGHTRAG_FASTTEXT_CB_FAILURE_THRESHOLD=5
export LIGHTRAG_FASTTEXT_CB_RECOVERY_TIMEOUT=60
export LIGHTRAG_FASTTEXT_ENABLE_RETRY_LOGIC=true
export LIGHTRAG_FASTTEXT_RETRY_MAX_ATTEMPTS=3
```

## Troubleshooting

### FastText Not Available

If FastText detection is not working:

1. Check if FastText library is installed:
   ```bash
   pip install fasttext
   ```

2. Check if model is downloaded:
   ```python
   service = LanguageDetectionService()
   print(service.is_fasttext_available())
   # Should return True
   ```

3. Check health status:
   ```python
   health = service.health_check()
   print(health['components']['fasttext_detector'])
   ```

### Circuit Breaker Open

If FastText is disabled due to circuit breaker:

```python
# Check circuit breaker status
metrics = service.get_performance_metrics()
print(metrics['circuit_breaker'])

# Wait for recovery timeout (default 60 seconds)
# Or reset manually:
service.reset_metrics()  # Resets circuit breaker
```

### Performance Issues

If detection is slow:

1. Check cache hit rates:
   ```python
   metrics = service.get_performance_metrics()
   print(metrics['fasttext_metrics']['cache_stats'])
   ```

2. Increase cache size:
   ```python
   service.update_configuration(cache_size=20000)
   ```

3. Use batch detection for multiple texts:
   ```python
   results = service.detect_batch(texts)  # Faster than individual calls
   ```

## Summary

Task 13.1 successfully wires all FastText language detection components together with:

✅ **Proper Component Integration**
- FastTextDetector, HybridDetectionStrategy, and LanguageDetectionService connected
- All dependencies properly injected
- Initialization order ensures correct setup

✅ **Configuration Management**
- Configuration flows through all layers
- Runtime updates supported for safe parameters
- Environment variable integration

✅ **Resource Management**
- Proper cleanup and reset functionality
- Memory-efficient caching with limits
- Automatic resource cleanup

✅ **Monitoring and Observability**
- Comprehensive metrics collection
- Health checks for all components
- Prometheus integration

✅ **Security and Resilience**
- Input sanitization and privacy protection
- Circuit breaker and retry logic
- Rate limiting for abuse prevention

✅ **Backward Compatibility**
- Drop-in replacement for LanguageDetector
- All original APIs preserved
- Seamless integration with existing code

The system is production-ready and fully integrated!
