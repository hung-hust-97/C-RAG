"""
Prompt management module with language detection and localization support.

This module provides the PromptManager class for retrieving and managing prompt templates
with automatic language detection and Vietnamese localization support.
"""

import html
import logging
import re
from typing import Optional
from lightrag.language_detector import LanguageDetector, SupportedLanguage

# Configure logger for this module
logger = logging.getLogger(__name__)


class PromptManager:
    """
    Manages prompt templates with language support and automatic detection.
    
    The PromptManager class provides methods to retrieve prompt templates in different
    languages (English and Vietnamese) based on explicit language specification or
    automatic language detection from input text.
    
    Vietnamese prompt variants are stored with a "_vi" suffix appended to the base
    prompt key. If a Vietnamese variant is not available, the system falls back to
    the English version.
    
    Attributes:
        prompts (dict): Reference to the PROMPTS dictionary containing all prompt templates
        detector (LanguageDetector): Language detector instance for automatic language detection
    
    Example:
        >>> from lightrag.prompt import PROMPTS
        >>> manager = PromptManager(PROMPTS)
        >>> 
        >>> # Explicit language specification
        >>> prompt = manager.get_prompt(
        ...     "entity_extraction_system_prompt",
        ...     language=SupportedLanguage.VIETNAMESE
        ... )
        >>> 
        >>> # Automatic language detection
        >>> text = "Trích xuất thực thể từ văn bản"
        >>> prompt = manager.get_prompt(
        ...     "entity_extraction_system_prompt",
        ...     auto_detect_text=text
        ... )
    """
    
    def __init__(self, prompts_dict: dict):
        """
        Initialize the PromptManager with a prompts dictionary.
        
        Args:
            prompts_dict: Dictionary containing all prompt templates. This should be
                         a reference to the PROMPTS dictionary from lightrag.prompt.
        
        Raises:
            ValueError: If prompts_dict is None or not a dictionary
        """
        if prompts_dict is None or not isinstance(prompts_dict, dict):
            raise ValueError("prompts_dict must be a non-None dictionary")
        
        self.prompts = prompts_dict
        self.detector = LanguageDetector()
    
    def get_prompt(
        self,
        key: str,
        language: Optional[SupportedLanguage] = None,
        auto_detect_text: Optional[str] = None
    ) -> str:
        """
        Get prompt template with language support.
        
        This method retrieves a prompt template based on the specified language or
        automatically detected language from input text. The language determination
        follows this priority:
        
        1. If `language` is explicitly provided, use that language
        2. If `auto_detect_text` is provided, detect language from the text
        3. Otherwise, default to English
        
        For Vietnamese prompts, the method appends "_vi" suffix to the key and
        attempts to retrieve the Vietnamese variant. If not found, it falls back
        to the English version with warning logging.
        
        Args:
            key: Prompt key (e.g., "entity_extraction_system_prompt")
            language: Explicit language override. If provided, this takes precedence
                     over auto-detection.
            auto_detect_text: Text to auto-detect language from. Only used if
                            language parameter is not provided.
        
        Returns:
            Localized prompt template string. Never returns None or empty string
            for valid prompt keys.
        
        Raises:
            KeyError: If the prompt key does not exist in the prompts dictionary
            ValueError: If language is provided but is not a valid SupportedLanguage
        
        Example:
            >>> manager = PromptManager(PROMPTS)
            >>> 
            >>> # Explicit Vietnamese
            >>> prompt = manager.get_prompt(
            ...     "rag_response",
            ...     language=SupportedLanguage.VIETNAMESE
            ... )
            >>> 
            >>> # Auto-detect from Vietnamese text
            >>> prompt = manager.get_prompt(
            ...     "rag_response",
            ...     auto_detect_text="Tóm tắt văn bản này"
            ... )
            >>> 
            >>> # Default to English
            >>> prompt = manager.get_prompt("rag_response")
        """
        # Validate that the base key exists (Requirement 8.3)
        if not isinstance(key, str) or not key:
            raise ValueError("Prompt key must be a non-empty string")
        
        if key not in self.prompts:
            raise KeyError(f"Prompt key '{key}' not found in prompts dictionary")
        
        # Validate language parameter if provided (Requirement 8.3)
        if language is not None and not isinstance(language, SupportedLanguage):
            raise ValueError(
                f"language must be a SupportedLanguage enum, got {type(language).__name__}. "
                f"Valid values are: {', '.join([lang.value for lang in SupportedLanguage])}"
            )
        
        # Step 1: Determine target language
        target_language: SupportedLanguage
        
        try:
            if language is not None:
                # Explicit language takes precedence
                target_language = language
            elif auto_detect_text is not None:
                # Sanitize auto-detect text before processing (Requirements 11.2, 11.3)
                sanitized_text = self._sanitize_input(auto_detect_text, "auto_detect_text")
                # Auto-detect from provided text
                target_language = self.detector.detect(sanitized_text)
            else:
                # Default to English
                target_language = SupportedLanguage.ENGLISH
        except Exception as e:
            # Handle any language detection errors gracefully (Requirement 8.5)
            logger.warning(f"Language detection failed for key '{key}': {e}. Falling back to English.")
            target_language = SupportedLanguage.ENGLISH
        
        # Step 2: Build localized prompt key
        if target_language == SupportedLanguage.VIETNAMESE:
            localized_key = f"{key}_vi"
        else:
            localized_key = key
        
        # Step 3: Retrieve prompt with fallback mechanism (Requirement 8.1)
        try:
            if localized_key in self.prompts:
                prompt = self.prompts[localized_key]
            else:
                # Fallback to English version with warning logging (Requirement 8.1)
                if target_language == SupportedLanguage.VIETNAMESE:
                    logger.warning(
                        f"Vietnamese prompt variant '{localized_key}' not found. "
                        f"Falling back to English prompt '{key}' for language support."
                    )
                prompt = self.prompts[key]
            
            # Validate that we got a non-empty prompt
            if not prompt or not isinstance(prompt, str):
                logger.warning(f"Prompt '{key}' is empty or invalid. Using fallback.")
                # If even the base prompt is invalid, return a minimal fallback
                return f"Error: Prompt '{key}' is not available."
            
            return prompt
            
        except Exception as e:
            # Handle any prompt retrieval errors gracefully (Requirement 8.5)
            logger.warning(f"Error retrieving prompt '{key}': {e}. Using fallback.")
            return f"Error: Prompt '{key}' is not available."
    
    def _sanitize_input(self, value: str, param_name: str) -> str:
        """
        Sanitize user-provided text to prevent prompt injection attacks.
        
        This method implements comprehensive input sanitization as required by
        Requirements 11.2 and 11.3. It escapes special characters that could
        be used for prompt injection while preserving legitimate content.
        
        Security measures implemented:
        - HTML/XML character escaping (prevents XSS-style attacks)
        - Prompt injection pattern detection and filtering
        - Delimiter escaping (prevents prompt structure manipulation)
        - Length validation (prevents DoS attacks)
        - Control character removal
        - Multiple newline normalization
        
        Args:
            value: The input string to sanitize
            param_name: Name of the parameter (for error reporting)
            
        Returns:
            Sanitized string safe for prompt injection
            
        Raises:
            ValueError: If input contains potentially malicious patterns or exceeds length limits
        """
        if not isinstance(value, str):
            return value
            
        # Check for extremely long strings that could cause DoS (Requirement 8.4)
        if len(value) > 1_000_000:  # 1MB limit
            raise ValueError(
                f"Parameter '{param_name}' exceeds maximum length of 1,000,000 characters"
            )
        
        # Detect and prevent common prompt injection patterns
        injection_patterns = [
            # Common prompt injection attempts
            r'(?i)(ignore\s+(?:previous|all|above)\s+(?:instructions?|prompts?))',
            r'(?i)(forget\s+(?:everything|all|previous))',
            r'(?i)(system\s*:\s*)',
            r'(?i)(assistant\s*:\s*)',
            r'(?i)(human\s*:\s*)',
            r'(?i)(user\s*:\s*)',
            # Role manipulation attempts
            r'(?i)(you\s+are\s+now\s+)',
            r'(?i)(act\s+as\s+)',
            r'(?i)(pretend\s+to\s+be\s+)',
            # Instruction override attempts
            r'(?i)(new\s+instructions?\s*:)',
            r'(?i)(override\s+)',
            r'(?i)(disregard\s+)',
        ]
        
        # Check for injection patterns
        for pattern in injection_patterns:
            if re.search(pattern, value):
                logger.warning(
                    f"Potential prompt injection detected in parameter '{param_name}': "
                    f"matches pattern {pattern}"
                )
                # Replace the suspicious content with safe placeholder
                value = re.sub(pattern, '[FILTERED]', value, flags=re.IGNORECASE)
        
        # Escape HTML/XML special characters to prevent injection
        # This handles <, >, &, ", ' which could be used in injection attempts
        value = html.escape(value, quote=True)
        
        # Escape common prompt delimiter patterns that could break prompt structure
        # These are commonly used in LLM prompts and could be exploited
        delimiter_escapes = {
            '---': '\\-\\-\\-',  # Common section delimiter
            '###': '\\#\\#\\#',  # Common header delimiter
            '```': '\\`\\`\\`',  # Code block delimiter
            '***': '\\*\\*\\*',  # Emphasis delimiter
            '===': '\\=\\=\\=',  # Another common delimiter
        }
        
        for delimiter, escaped in delimiter_escapes.items():
            value = value.replace(delimiter, escaped)
        
        # Escape newlines that could break prompt structure
        # Replace multiple consecutive newlines with single newline to prevent
        # prompt structure manipulation
        value = re.sub(r'\n{3,}', '\n\n', value)
        
        # Remove or escape null bytes and other control characters
        # that could cause parsing issues
        value = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', value)
        
        return value

    def format_prompt(
        self,
        key: str,
        prompt_language: SupportedLanguage,
        **kwargs
    ) -> str:
        """
        Format prompt template with parameter substitution and validation.
        
        This method retrieves a prompt template using get_prompt() and then
        formats it with the provided keyword arguments. It includes parameter
        type validation to prevent injection attacks and ensure type safety.
        
        The method validates that all parameters are of safe types (str, int,
        float, bool, list) before formatting. This prevents potential security
        issues from malicious parameter values.
        
        Args:
            key: Prompt key (e.g., "entity_extraction_system_prompt")
            prompt_language: Language for the prompt template selection (ENGLISH or VIETNAMESE)
            **kwargs: Keyword arguments for prompt parameter substitution.
                     Common parameters include:
                     - entity_types: str
                     - tuple_delimiter: str
                     - completion_delimiter: str
                     - input_text: str
                     - examples: str or list
                     - language: str (the language name for {language} placeholder, e.g., "English" or "Vietnamese")
                     - etc.
        
        Returns:
            Formatted prompt string with all parameters substituted
        
        Raises:
            KeyError: If the prompt key does not exist
            ValueError: If any parameter has an invalid type, if required
                       parameters are missing for formatting, or if prompt_language is invalid
            TypeError: If parameter types are not safe for string formatting
        
        Example:
            >>> manager = PromptManager(PROMPTS)
            >>> formatted = manager.format_prompt(
            ...     "entity_extraction_user_prompt",
            ...     prompt_language=SupportedLanguage.VIETNAMESE,
            ...     entity_types="Person, Organization, Location",
            ...     input_text="Nguyễn Văn A làm việc tại công ty ABC",
            ...     tuple_delimiter="<|#|>",
            ...     completion_delimiter="<|COMPLETE|>",
            ...     language="Vietnamese"
            ... )
        """
        # Validate prompt_language parameter (Requirement 8.3)
        if not isinstance(prompt_language, SupportedLanguage):
            raise ValueError(
                f"prompt_language must be a SupportedLanguage enum, got {type(prompt_language).__name__}. "
                f"Valid values are: {', '.join([lang.value for lang in SupportedLanguage])}"
            )
        
        try:
            # Step 1: Retrieve the prompt template
            prompt_template = self.get_prompt(key, language=prompt_language)
            
            # Step 2: Validate and sanitize parameters for security (Requirements 11.2, 11.3, 11.5)
            # Only allow safe types to prevent injection attacks
            safe_types = (str, int, float, bool, list, type(None))
            sanitized_kwargs = {}
            
            for param_name, param_value in kwargs.items():
                # Type validation (Requirement 11.5)
                if not isinstance(param_value, safe_types):
                    raise TypeError(
                        f"Parameter '{param_name}' has invalid type {type(param_value).__name__}. "
                        f"Only str, int, float, bool, list, and None are allowed for security reasons."
                    )
                
                # Sanitize string parameters to prevent injection (Requirements 11.2, 11.3)
                if isinstance(param_value, str):
                    sanitized_kwargs[param_name] = self._sanitize_input(param_value, param_name)
                elif isinstance(param_value, list):
                    # Sanitize list elements if they are strings
                    sanitized_list = []
                    for i, item in enumerate(param_value):
                        if isinstance(item, str):
                            sanitized_list.append(self._sanitize_input(item, f"{param_name}[{i}]"))
                        elif isinstance(item, safe_types):
                            sanitized_list.append(item)
                        else:
                            raise TypeError(
                                f"List parameter '{param_name}[{i}]' contains invalid type "
                                f"{type(item).__name__}. Only safe types are allowed."
                            )
                    sanitized_kwargs[param_name] = sanitized_list
                else:
                    # Non-string types pass through unchanged after type validation
                    sanitized_kwargs[param_name] = param_value
            
            # Step 3: Format the prompt with validated and sanitized parameters
            formatted_prompt = prompt_template.format(**sanitized_kwargs)
            
            # Validate the formatted result
            if not formatted_prompt or not isinstance(formatted_prompt, str):
                logger.warning(f"Formatted prompt for key '{key}' is empty or invalid")
                return f"Error: Unable to format prompt '{key}'"
            
            return formatted_prompt
            
        except KeyError as e:
            # Missing required parameter
            error_msg = f"Missing required parameter for prompt '{key}': {e}"
            logger.warning(error_msg)
            raise ValueError(error_msg) from e
            
        except (ValueError, TypeError) as e:
            # Re-raise validation errors
            logger.warning(f"Parameter validation failed for prompt '{key}': {e}")
            raise
            
        except (IndexError, AttributeError) as e:
            # Invalid format string or parameter mismatch
            error_msg = f"Error formatting prompt '{key}': {e}"
            logger.warning(error_msg)
            raise ValueError(error_msg) from e
            
        except Exception as e:
            # Handle any unexpected errors gracefully (Requirement 8.5)
            error_msg = f"Unexpected error formatting prompt '{key}': {e}"
            logger.warning(error_msg)
            # Return a fallback instead of raising to ensure system continues operation
            return f"Error: Unable to format prompt '{key}' due to unexpected error"
