"""
Security utilities for API client operations.

This module provides security-related functionality including API key validation,
input sanitization, and suspicious pattern detection.
"""

import re
import time
import hashlib
import html
import logging
from typing import Dict, List
from soap_kg.config import Config

logger = logging.getLogger(__name__)


class SecurityValidator:
    """Security validation and sanitization utilities."""
    
    def __init__(self):
        self.suspicious_patterns = [
            r'<script[^>]*>.*?</script>',  # Script injection
            r'javascript:',  # JavaScript protocol
            r'data:text/html',  # Data URL HTML
            r'\\x[0-9a-fA-F]{2}',  # Hex encoding
            r'%[0-9a-fA-F]{2}',  # URL encoding of control chars
            r'\\u[0-9a-fA-F]{4}',  # Unicode escape sequences
            r'eval\s*\(',  # eval function
            r'exec\s*\(',  # exec function
            r'__import__\s*\(',  # Python import
            r'system\s*\(',  # System calls
        ]
    
    def validate_api_key(self, api_key: str) -> None:
        """Validate API key format and basic security requirements."""
        if not api_key or not Config.ENABLE_API_KEY_VALIDATION:
            return
            
        # Check for minimum length
        if len(api_key) < Config.MIN_API_KEY_LENGTH:
            logger.warning(f"API key appears to be too short (length: {len(api_key)}, minimum: {Config.MIN_API_KEY_LENGTH})")
            
        # Check for obvious test/placeholder values
        if not Config.ALLOW_TEST_API_KEYS:
            test_patterns = [
                r'test', r'demo', r'example', r'placeholder', r'your.*key.*here',
                r'sk-.*test', r'fake', r'dummy', r'sample'
            ]
            
            for pattern in test_patterns:
                if re.search(pattern, api_key.lower()):
                    logger.warning("API key appears to be a test/placeholder value - this may cause authentication failures")
                    break
                    
        # Log successful validation without exposing the key
        logger.info(f"API key validated (length: {len(api_key)}, masked: {self.mask_api_key(api_key)})")
    
    def mask_api_key(self, api_key: str) -> str:
        """Safely mask API key for logging purposes."""
        if not api_key:
            return "None"
        if len(api_key) <= 8:
            return "*" * len(api_key)
        return f"{api_key[:4]}{'*' * (len(api_key) - 8)}{api_key[-4:]}"
    
    def mask_sensitive_data(self, text: str) -> str:
        """Remove sensitive data from text for safe logging."""
        if not text or not Config.MASK_SENSITIVE_DATA_IN_LOGS:
            return text
            
        # Mask potential API keys in the text
        # Pattern for Bearer tokens
        text = re.sub(r'Bearer\s+[A-Za-z0-9\-_]{20,}', 'Bearer [MASKED]', text)
        
        # Pattern for sk-* style keys (OpenAI/OpenRouter format)
        text = re.sub(r'sk-[A-Za-z0-9\-_]{8,}', 'sk-[MASKED]', text)
        
        # Pattern for other potential API keys (long alphanumeric strings)
        text = re.sub(r'\b[A-Za-z0-9\-_]{32,}\b', '[MASKED_KEY]', text)
        
        # Mask Authorization headers
        text = re.sub(r'"Authorization":\s*"[^"]*"', '"Authorization": "[MASKED]"', text)
        
        # Mask any remaining long tokens that might be API keys
        text = re.sub(r'\b[A-Za-z0-9\-_.]{50,}\b', '[MASKED_LONG_TOKEN]', text)
        
        return text
    
    def sanitize_input_text(self, text: str) -> str:
        """Sanitize input text to prevent injection attacks."""
        if not text or not Config.SANITIZE_INPUT_TEXT:
            return text
            
        # Remove or escape potentially dangerous characters
        sanitized = text.strip()
        
        # HTML escape to prevent HTML injection
        sanitized = html.escape(sanitized, quote=True)
        
        # Remove null bytes and other control characters
        sanitized = ''.join(char for char in sanitized if ord(char) >= 32 or char in '\n\r\t')
        
        # Limit length to prevent oversized payloads
        if len(sanitized) > Config.MAX_PROMPT_LENGTH:
            logger.warning(f"Input text truncated from {len(sanitized)} to {Config.MAX_PROMPT_LENGTH} characters")
            sanitized = sanitized[:Config.MAX_PROMPT_LENGTH]
            
        return sanitized
    
    def detect_suspicious_patterns(self, text: str) -> bool:
        """Detect suspicious patterns that might indicate injection attacks."""
        if not text or not Config.BLOCK_SUSPICIOUS_PATTERNS:
            return False
            
        for pattern in self.suspicious_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                logger.warning(f"Suspicious pattern detected in input: {pattern}")
                if Config.LOG_SECURITY_EVENTS:
                    # Create a hash of the suspicious text for logging without exposing content
                    text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
                    logger.warning(f"SECURITY_EVENT: Suspicious pattern blocked (hash: {text_hash})")
                return True
                
        return False
    
    def log_security_event(self, event_type: str, details: str):
        """Log security events for monitoring."""
        if Config.LOG_SECURITY_EVENTS:
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            logger.warning(f"SECURITY_EVENT [{timestamp}] {event_type}: {details}")


class RequestValidator:
    """Request validation utilities."""
    
    @staticmethod
    def validate_request_size(payload: Dict) -> bool:
        """Validate request payload size."""
        try:
            import json
            payload_json = json.dumps(payload)
            payload_size = len(payload_json.encode('utf-8'))
            
            if payload_size > Config.MAX_REQUEST_SIZE_BYTES:
                logger.error(f"Request payload too large: {payload_size} bytes (max: {Config.MAX_REQUEST_SIZE_BYTES})")
                return False
                
            logger.debug(f"Request payload size: {payload_size} bytes")
            return True
            
        except Exception as e:
            logger.error(f"Error validating request size: {e}")
            return False