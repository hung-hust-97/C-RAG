# Security and Privacy Features

This document describes the comprehensive security and privacy protections implemented in the FastText language detection system (Task 12.1).

## Overview

The FastText language detection system implements multiple layers of security and privacy protections to ensure safe operation and compliance with data privacy regulations (GDPR, CCPA).

## Security Features

### 1. Input Sanitization (Requirement 10.2)

**Purpose**: Prevent injection attacks and ensure safe text processing.

**Implementation**: `InputSanitizer` class in `security.py`

**Features**:
- **Length Validation**: Rejects text exceeding maximum length (default: 100,000 characters) to prevent DoS attacks
- **Null Byte Removal**: Removes null bytes (`\x00`) to prevent path traversal and injection attacks
- **Control Character Filtering**: Removes dangerous control characters while preserving common whitespace (tabs, newlines)
- **Script Tag Detection**: Detects and removes/rejects dangerous patterns like `<script>`, `javascript:`, event handlers
- **Unicode Normalization**: Normalizes text to NFC form to prevent homograph attacks
- **Strict Mode**: Optional strict mode that rejects (rather than sanitizes) dangerous patterns

**Usage**:
```python
from lightrag.language_detection.security import InputSanitizer

sanitizer = InputSanitizer(max_length=100000)

# Normal mode - sanitizes dangerous content
safe_text = sanitizer.sanitize(text, strict=False)

# Strict mode - rejects dangerous content
try:
    safe_text = sanitizer.sanitize(text, strict=True)
except ValueError as e:
    print(f"Dangerous content detected: {e}")
```

### 2. Rate Limiting (Requirement 10.5)

**Purpose**: Prevent abuse of detection services through rate limiting.

**Implementation**: `RateLimiter` class in `security.py`

**Features**:
- **Sliding Window Algorithm**: Tracks requests per client over a configurable time window
- **Burst Allowance**: Allows short bursts above the base limit
- **Per-Client Tracking**: Separate limits for each client (identified by client_id)
- **Thread-Safe**: Safe for concurrent access
- **Configurable**: All parameters configurable via `RateLimitConfig`
- **Statistics**: Provides detailed statistics for monitoring

**Configuration**:
```python
from lightrag.language_detection.security import RateLimiter, RateLimitConfig

config = RateLimitConfig(
    max_requests=100,      # Maximum requests per window
    window_seconds=60,     # Time window in seconds
    enabled=True,          # Enable/disable rate limiting
    burst_allowance=10     # Additional burst requests allowed
)

limiter = RateLimiter(config)
```

**Usage**:
```python
# Check rate limit before processing
allowed, retry_after = limiter.check_rate_limit(client_id="user_123")

if not allowed:
    raise ValueError(f"Rate limit exceeded. Retry after {retry_after} seconds.")

# Process request...
```

**Integration with Service**:
```python
from lightrag.language_detection.service import LanguageDetectionService

service = LanguageDetectionService()

# Detect with rate limiting
try:
    language = service.detect_language(text, client_id="user_123")
except ValueError as e:
    print(f"Rate limit error: {e}")
```

### 3. Secure Model Downloads (Requirements 10.3, 10.4)

**Purpose**: Ensure model files are downloaded securely and verified for integrity.

**Implementation**: `SecureDownloader` class in `security.py`, integrated into `FastTextModelManager`

**Features**:
- **HTTPS Enforcement**: Only allows HTTPS URLs for model downloads
- **URL Validation**: Rejects localhost, private IPs, and malformed URLs
- **Checksum Verification**: Verifies file integrity using SHA-256 checksums
- **Constant-Time Comparison**: Uses `hmac.compare_digest` to prevent timing attacks
- **Atomic Updates**: Downloads to temporary file, verifies, then moves atomically

**URL Validation Rules**:
- ✅ HTTPS URLs from public domains
- ❌ HTTP URLs (insecure)
- ❌ Localhost URLs (127.0.0.1, localhost, ::1)
- ❌ Private IP ranges (10.x.x.x, 172.16.x.x, 192.168.x.x)
- ❌ URLs without hostname

**Usage**:
```python
from lightrag.language_detection.security import SecureDownloader

downloader = SecureDownloader()

# Validate URL before download
if not downloader.validate_url(url):
    raise ValueError("Invalid or insecure URL")

# Verify downloaded file
if not downloader.verify_checksum(file_path, expected_checksum, algorithm='sha256'):
    raise ValueError("Checksum verification failed")
```

**Automatic Integration**:
The `FastTextModelManager` automatically uses `SecureDownloader` for all model downloads:
```python
from lightrag.language_detection.model_manager import FastTextModelManager

manager = FastTextModelManager()
# Downloads are automatically secured with HTTPS and checksum verification
model_path = manager.ensure_model_available()
```

## Privacy Features

### 4. No Text Content Logging (Requirement 10.1)

**Purpose**: Ensure user text content is never logged or persisted, complying with GDPR/CCPA.

**Implementation**: `PrivacyFilter` class in `security.py`, integrated throughout the system

**Features**:
- **Privacy-Safe Logging**: Logs only metadata (length, hash, word count) instead of actual text
- **Text Hashing**: Uses SHA-256 hashes for cache keys and logging
- **Log Message Sanitization**: Automatically redacts text content from log messages
- **No Persistence**: Detection results are cached by hash, not by text content

**Privacy-Safe Logging**:
```python
from lightrag.language_detection.security import PrivacyFilter

# Get safe logging information
log_info = PrivacyFilter.get_safe_log_info(text)
# Returns: {'text_hash': '...', 'length': 123, 'is_empty': False, ...}

# Sanitize log messages
message = "Processing text='sensitive content' for detection"
safe_message = PrivacyFilter.sanitize_log_message(message)
# Returns: "Processing text=[REDACTED] for detection"
```

**Automatic Integration**:
All detection methods automatically use privacy-safe logging:
```python
from lightrag.language_detection.fasttext_detector import FastTextDetector

detector = FastTextDetector()
# Text content is never logged - only metadata
language, confidence = detector.detect(text)
```

### 5. Secure Memory Handling (Requirement 10.6)

**Purpose**: Clear sensitive data from memory after processing.

**Implementation**: Integrated into all detection components

**Features**:
- **Cache TTL**: Detection results expire after configurable TTL (default: 1 hour)
- **Cache Size Limits**: LRU eviction prevents unbounded memory growth
- **Cleanup Methods**: Explicit cache clearing methods available
- **No Text Persistence**: Only hashes are stored in cache, not actual text

## Configuration

### Environment Variables

Security features can be configured via environment variables:

```bash
# Rate limiting (not yet exposed, but can be added)
export LIGHTRAG_FASTTEXT_RATE_LIMIT_ENABLED=true
export LIGHTRAG_FASTTEXT_RATE_LIMIT_MAX_REQUESTS=100
export LIGHTRAG_FASTTEXT_RATE_LIMIT_WINDOW_SECONDS=60

# Input validation
export LIGHTRAG_FASTTEXT_MAX_TEXT_LENGTH=100000

# Model download security
export LIGHTRAG_FASTTEXT_MODEL_URL=https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
export LIGHTRAG_FASTTEXT_MODEL_CHECKSUM=7e69ec5451bc261cc7844e49e4792a85d7f09c06789ec800fc4a44aec362764e
```

### Programmatic Configuration

```python
from lightrag.language_detection.config import FastTextConfig
from lightrag.language_detection.service import LanguageDetectionService

# Create custom configuration
config = FastTextConfig(
    max_text_length=50000,  # Stricter length limit
    model_url="https://example.com/model.bin",
    model_checksum="abc123...",
)

# Initialize service with custom config
service = LanguageDetectionService(config=config)
```

## Monitoring and Auditing

### Rate Limiter Statistics

```python
from lightrag.language_detection.service import LanguageDetectionService

service = LanguageDetectionService()

# Get rate limiter statistics
stats = service.rate_limiter.get_stats()
print(f"Total requests: {stats['total_requests']}")
print(f"Blocked requests: {stats['total_blocked']}")
print(f"Block rate: {stats['block_rate_percent']}%")
```

### Security Audit Logs

All security events are logged with appropriate severity levels:
- **INFO**: Normal operations (model downloads, successful validations)
- **WARNING**: Suspicious activity (rate limits exceeded, sanitization applied)
- **ERROR**: Security violations (invalid URLs, checksum failures, injection attempts)

Example log entries:
```
INFO: SecureDownloader initialized with HTTPS enforcement
WARNING: Rate limit exceeded for client [hash], retry after 45s
ERROR: Non-HTTPS URL rejected: http://example.com/model.bin
WARNING: Dangerous pattern detected in input: <script>
INFO: Checksum verification passed for /path/to/model.bin
```

## Compliance

### GDPR Compliance

- ✅ **Data Minimization**: Only necessary metadata is logged, not actual text content
- ✅ **Purpose Limitation**: Text is processed only for language detection, not stored
- ✅ **Storage Limitation**: Cache entries expire after TTL, no long-term storage
- ✅ **Integrity and Confidentiality**: Input sanitization and secure processing

### CCPA Compliance

- ✅ **No Sale of Data**: Text content is never persisted or shared
- ✅ **Right to Deletion**: Cache can be cleared on demand
- ✅ **Transparency**: Clear documentation of data handling practices

## Security Best Practices

### For Developers

1. **Always use the service layer**: Use `LanguageDetectionService` which includes all security protections
2. **Provide client_id for rate limiting**: Pass client identifiers when available
3. **Handle rate limit exceptions**: Implement proper error handling for rate limit violations
4. **Monitor security metrics**: Regularly check rate limiter and security statistics
5. **Keep checksums updated**: Update model checksums when upgrading models

### For Operators

1. **Configure appropriate rate limits**: Adjust based on expected load and abuse patterns
2. **Monitor security logs**: Watch for patterns of abuse or attacks
3. **Regular security audits**: Review security configurations and logs periodically
4. **Update dependencies**: Keep FastText and security libraries up to date
5. **Use HTTPS only**: Never configure HTTP URLs for model downloads

## Testing

Comprehensive security tests are provided in `test_security.py`:

```bash
# Run security tests
python -m pytest lightrag/language_detection/test_security.py -v

# Run specific test class
python -m pytest lightrag/language_detection/test_security.py::TestInputSanitizer -v
```

Test coverage includes:
- Input sanitization (10 tests)
- Rate limiting (7 tests)
- Secure downloads (7 tests)
- Privacy filtering (5 tests)

## Threat Model

### Threats Mitigated

1. **Injection Attacks**: Input sanitization prevents script injection, SQL injection, and command injection
2. **DoS Attacks**: Rate limiting and length validation prevent resource exhaustion
3. **Man-in-the-Middle**: HTTPS enforcement prevents model tampering during download
4. **Data Leakage**: Privacy filtering prevents sensitive text from appearing in logs
5. **Timing Attacks**: Constant-time checksum comparison prevents timing-based attacks

### Residual Risks

1. **Application-Level DoS**: Rate limiting is per-client; distributed attacks may still succeed
2. **Zero-Day Vulnerabilities**: Unknown vulnerabilities in dependencies
3. **Insider Threats**: Operators with system access can bypass protections

### Mitigation Recommendations

1. **Deploy behind WAF**: Use Web Application Firewall for additional protection
2. **Implement IP-based rate limiting**: Add network-level rate limiting
3. **Regular security updates**: Keep all dependencies updated
4. **Access controls**: Implement strict access controls for system operators
5. **Security monitoring**: Deploy intrusion detection and monitoring systems

## References

- **Requirements**: See `requirements.md` Section 10 (Security and Privacy)
- **Design**: See `design.md` Security and Privacy sections
- **Implementation**: See `security.py` for complete implementation
- **Tests**: See `test_security.py` for test coverage
