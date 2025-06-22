"""
Core API client for making HTTP requests to OpenRouter.

This module handles the low-level HTTP communication, SSL configuration,
retry logic, and basic request/response handling.
"""

import requests
import urllib3
import time
import logging
from typing import Dict, List, Optional
from soap_kg.config import Config
from soap_kg.utils.security import SecurityValidator, RequestValidator

logger = logging.getLogger(__name__)


class OpenRouterApiClient:
    """Core API client for OpenRouter HTTP requests."""
    
    def __init__(self, api_key: str = None, model: str = None, base_url: str = None):
        self.api_key = api_key or Config.OPENROUTER_API_KEY
        self.base_url = base_url or Config.OPENROUTER_BASE_URL
        self.model = model or Config.DEFAULT_MODEL
        
        # Initialize security validator
        self.security_validator = SecurityValidator()
        
        # Validate API key security
        if self.api_key:
            self.security_validator.validate_api_key(self.api_key)
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/jafffy/medical_kg",
            "X-Title": "SOAP Knowledge Graph Generator"
        }
    
    def make_request(self, messages: List[Dict], max_tokens: int = 1000, 
                    temperature: float = 0.1, max_retries: int = 2) -> Optional[str]:
        """Make a request to OpenRouter API with retry logic."""
        if not self.api_key:
            logger.error("OpenRouter API key not provided")
            return None
            
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        for attempt in range(max_retries + 1):
            try:
                # Validate payload size before sending
                if not RequestValidator.validate_request_size(payload):
                    logger.error("Request payload exceeds size limits")
                    self.security_validator.log_security_event(
                        "OVERSIZED_REQUEST", 
                        f"Payload size exceeds {Config.MAX_REQUEST_SIZE_BYTES} bytes"
                    )
                    return None
                
                # Configure SSL settings
                verify_ssl = Config.VERIFY_SSL_CERTIFICATES
                if not verify_ssl:
                    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                    logger.warning("SSL certificate verification is disabled")
                
                # Log request if enabled
                if Config.ENABLE_REQUEST_LOGGING:
                    safe_payload = {**payload}
                    safe_payload['messages'] = '[MASKED]'  # Don't log actual content
                    logger.debug(f"Making API request: {safe_payload}")
                
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=Config.REQUEST_TIMEOUT,
                    verify=verify_ssl
                )
                
                # Validate response size
                response_size = len(response.content)
                if response_size > Config.MAX_RESPONSE_SIZE_BYTES:
                    logger.error(f"Response too large: {response_size} bytes (max: {Config.MAX_RESPONSE_SIZE_BYTES})")
                    self.security_validator.log_security_event(
                        "OVERSIZED_RESPONSE", 
                        f"Response size {response_size} exceeds limit"
                    )
                    return None
                
                if response.status_code == 200:
                    result = response.json()
                    content = result['choices'][0]['message']['content']
                    if content and content.strip():
                        return content
                    else:
                        logger.warning(f"Empty response from OpenRouter on attempt {attempt + 1}")
                        continue
                elif response.status_code == 429:  # Rate limit
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(f"Rate limited, waiting {wait_time}s before retry {attempt + 1}")
                    time.sleep(wait_time)
                    continue
                else:
                    # Mask sensitive data in error logs
                    safe_response_text = self.security_validator.mask_sensitive_data(response.text)
                    logger.error(f"OpenRouter API error: {response.status_code} - {safe_response_text}")
                    if attempt < max_retries:
                        continue
                    return None
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Request timeout on attempt {attempt + 1}")
                if attempt < max_retries:
                    continue
                return None
            except Exception as e:
                # Mask sensitive data in exception logs
                safe_error_message = self.security_validator.mask_sensitive_data(str(e))
                logger.error(f"Error making OpenRouter request on attempt {attempt + 1}: {safe_error_message}")
                if attempt < max_retries:
                    continue
                return None
        
        return None
    
    def check_api_status(self) -> bool:
        """Check if the API is accessible and properly configured."""
        if not self.api_key:
            logger.error("API key not configured")
            return False
        
        # Simple test request
        test_messages = [{"role": "user", "content": "test"}]
        try:
            response = self.make_request(test_messages, max_tokens=1, max_retries=1)
            return response is not None
        except Exception as e:
            logger.error(f"API status check failed: {e}")
            return False
    
    def get_model_info(self) -> Optional[Dict]:
        """Get information about the current model."""
        return {
            "model": self.model,
            "base_url": self.base_url,
            "api_key_configured": bool(self.api_key),
            "api_key_masked": self.security_validator.mask_api_key(self.api_key) if self.api_key else None
        }