"""
Example usage of the enhanced LanguageDetectionService.

This example demonstrates how to use the LanguageDetectionService as a drop-in
replacement for the existing LanguageDetector while taking advantage of the
enhanced FastText capabilities.
"""

from lightrag.language_detector import SupportedLanguage
from lightrag.language_detection import LanguageDetectionService, FastTextConfig


def example_basic_usage():
    """Example 1: Basic usage maintaining backward compatibility."""
    print("=" * 60)
    print("Example 1: Basic Usage (Backward Compatible)")
    print("=" * 60)
    
    # Initialize service with default configuration
    service = LanguageDetectionService()
    
    # Use the same API as LanguageDetector
    english_text = "This is an English sentence."
    result = service.detect_language(english_text)
    print(f"Text: {english_text}")
    print(f"Detected: {result.value}\n")
    
    vietnamese_text = "Đây là một câu tiếng Việt."
    result = service.detect_language(vietnamese_text)
    print(f"Text: {vietnamese_text}")
    print(f"Detected: {result.value}\n")
    
    # Use convenience method
    print(f"Is Vietnamese? {service.is_vietnamese(vietnamese_text)}")
    print(f"Is Vietnamese? {service.is_vietnamese(english_text)}\n")


def example_detailed_detection():
    """Example 2: Using enhanced detection with detailed results."""
    print("=" * 60)
    print("Example 2: Enhanced Detection with Details")
    print("=" * 60)
    
    service = LanguageDetectionService()
    
    text = "This is a test sentence for language detection."
    result = service.get_detection_details(text)
    
    print(f"Text: {text}")
    print(f"Language: {result.language.value}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Method Used: {result.method_used}")
    print(f"Processing Time: {result.processing_time_ms:.2f}ms")
    print(f"Text Length: {result.text_length}")
    print(f"Cached: {result.cached}")
    print(f"FastText Available: {result.fasttext_available}")
    print(f"Unicode Fallback Used: {result.unicode_fallback_used}")
    print(f"Hybrid Strategy Applied: {result.hybrid_strategy_applied}\n")


def example_batch_detection():
    """Example 3: Batch detection for multiple texts."""
    print("=" * 60)
    print("Example 3: Batch Detection")
    print("=" * 60)
    
    service = LanguageDetectionService()
    
    texts = [
        "English text",
        "Tiếng Việt",
        "Another English sentence",
        "Câu tiếng Việt khác",
        "Mixed English và Việt"
    ]
    
    results = service.detect_batch(texts)
    
    print("Batch Detection Results:")
    for text, result in zip(texts, results):
        print(f"  '{text}' -> {result.value}")
    print()


def example_health_monitoring():
    """Example 4: Health check and monitoring."""
    print("=" * 60)
    print("Example 4: Health Check and Monitoring")
    print("=" * 60)
    
    service = LanguageDetectionService()
    
    # Perform some detections
    service.detect_language("Test")
    service.detect_language("Thử nghiệm")
    
    # Check health status
    health = service.health_check()
    print(f"Service Status: {health['status']}")
    print(f"Service Available: {health['service_available']}")
    print(f"FastText Available: {health['components']['fasttext_detector']['available']}")
    print(f"Unicode Available: {health['components']['unicode_detector']['available']}")
    print()
    
    # Get performance metrics
    metrics = service.get_performance_metrics()
    print(f"Total Detections: {metrics['total_detections']}")
    print(f"Average Processing Time: {metrics['average_processing_time_ms']:.2f}ms")
    print(f"Method Usage: {metrics['method_usage_counts']}")
    print()


def example_custom_configuration():
    """Example 5: Using custom configuration."""
    print("=" * 60)
    print("Example 5: Custom Configuration")
    print("=" * 60)
    
    # Create custom configuration
    config = FastTextConfig()
    config.enabled = True
    config.cache_size = 5000
    config.vietnamese_threshold = 0.75
    config.log_detection_method = True
    
    # Initialize service with custom config
    service = LanguageDetectionService(config=config)
    
    text = "Đây là văn bản tiếng Việt để kiểm tra."
    result = service.detect_language(text)
    
    print(f"Text: {text}")
    print(f"Detected: {result.value}")
    print("Configuration:")
    print(f"  Cache Size: {service.config.cache_size}")
    print(f"  Vietnamese Threshold: {service.config.vietnamese_threshold}")
    print(f"  FastText Enabled: {service.config.enabled}")
    print()


def example_integration_with_prompt_manager():
    """Example 6: Integration with PromptManager (conceptual)."""
    print("=" * 60)
    print("Example 6: Integration with PromptManager")
    print("=" * 60)
    
    # This demonstrates how LanguageDetectionService can be used
    # as a drop-in replacement for LanguageDetector in PromptManager
    
    service = LanguageDetectionService()
    
    # Simulate PromptManager usage
    user_input = "Trích xuất thực thể từ văn bản này"
    
    # Detect language from user input
    detected_language = service.detect_language(user_input)
    
    print(f"User Input: {user_input}")
    print(f"Detected Language: {detected_language.value}")
    print("Would select prompt variant: entity_extraction_system_prompt_vi")
    print()
    
    # The PromptManager would then use this to select the appropriate prompt:
    # prompt_key = "entity_extraction_system_prompt"
    # if detected_language == SupportedLanguage.VIETNAMESE:
    #     prompt_key = f"{prompt_key}_vi"
    # prompt = prompts[prompt_key]


def example_fasttext_disabled():
    """Example 7: Service with FastText disabled (Unicode-only mode)."""
    print("=" * 60)
    print("Example 7: Unicode-Only Mode (FastText Disabled)")
    print("=" * 60)
    
    # Create configuration with FastText disabled
    config = FastTextConfig()
    config.enabled = False
    
    service = LanguageDetectionService(config=config)
    
    text = "Đây là tiếng Việt"
    result = service.detect_language(text)
    
    print(f"Text: {text}")
    print(f"Detected: {result.value}")
    print(f"FastText Available: {service.is_fasttext_available()}")
    print("Using Unicode detection only")
    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("LanguageDetectionService Usage Examples")
    print("=" * 60 + "\n")
    
    try:
        example_basic_usage()
        example_detailed_detection()
        example_batch_detection()
        example_health_monitoring()
        example_custom_configuration()
        example_integration_with_prompt_manager()
        example_fasttext_disabled()
        
        print("=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
