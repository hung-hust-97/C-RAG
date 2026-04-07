"""
Logging configuration for FastText language detection components.

This module sets up dedicated logging for the FastText language detection system
while integrating with LightRAG's existing logging infrastructure.
"""

import logging

# Create a dedicated logger for FastText components
fasttext_logger = logging.getLogger("lightrag.fasttext")
fasttext_logger.setLevel(logging.INFO)
fasttext_logger.propagate = True  # Allow propagation to parent lightrag logger

# Set up basic console handler if no handlers exist
if not fasttext_logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    fasttext_logger.addHandler(console_handler)

def setup_fasttext_logging(level: str = "INFO", enable_file_logging: bool = True):
    """
    Set up logging for FastText language detection components.
    
    This function configures a dedicated logger for FastText components that
    integrates with LightRAG's logging infrastructure. The logger follows
    the same patterns as other LightRAG components.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        enable_file_logging: Whether to enable logging to file
    """
    # Set log level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    fasttext_logger.setLevel(numeric_level)
    
    fasttext_logger.info("FastText logging configured successfully")


def get_fasttext_logger(component_name: str = None) -> logging.Logger:
    """
    Get a logger instance for FastText components.
    
    Args:
        component_name: Optional component name to append to logger name
        
    Returns:
        Logger instance for the specified component
    """
    if component_name:
        logger_name = f"lightrag.fasttext.{component_name}"
        return logging.getLogger(logger_name)
    else:
        return fasttext_logger


# Initialize FastText logging on module import
setup_fasttext_logging()

# Export the main logger for convenience
logger = fasttext_logger