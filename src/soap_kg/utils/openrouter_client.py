"""
Refactored OpenRouter client using modular architecture.

This is a backward-compatible wrapper that maintains the original API while
leveraging the new modular components for better maintainability.
"""

import logging
from typing import Dict, List, Optional
from soap_kg.utils.medical_nlp_client import MedicalNLPClient

logger = logging.getLogger(__name__)


class OpenRouterClient:
    """
    Backward-compatible OpenRouter client.
    
    This class provides the same interface as the original monolithic client
    but uses the new modular architecture under the hood.
    """
    
    def __init__(self, api_key: str = None, model: str = None):
        # Initialize the modular components
        self.medical_nlp_client = MedicalNLPClient(api_key=api_key, model=model)
        self.api_client = self.medical_nlp_client.api_client
        self.security_validator = self.medical_nlp_client.security_validator
        self.response_parser = self.medical_nlp_client.response_parser
        
        # Expose core properties for backward compatibility
        self.api_key = self.api_client.api_key
        self.base_url = self.api_client.base_url
        self.model = self.api_client.model
        self.headers = self.api_client.headers
    
    # Backward compatibility methods - delegate to modular components
    
    def _validate_api_key(self) -> None:
        """Validate API key format and basic security requirements."""
        if self.api_key:
            self.security_validator.validate_api_key(self.api_key)
    
    def _mask_api_key(self, api_key: str) -> str:
        """Safely mask API key for logging purposes."""
        return self.security_validator.mask_api_key(api_key)
    
    def _mask_sensitive_data(self, text: str) -> str:
        """Remove sensitive data from text for safe logging."""
        return self.security_validator.mask_sensitive_data(text)
    
    def _sanitize_input_text(self, text: str) -> str:
        """Sanitize input text to prevent injection attacks."""
        return self.security_validator.sanitize_input_text(text)
    
    def _validate_request_size(self, payload: Dict) -> bool:
        """Validate request payload size."""
        from soap_kg.utils.security import RequestValidator
        return RequestValidator.validate_request_size(payload)
    
    def _detect_suspicious_patterns(self, text: str) -> bool:
        """Detect suspicious patterns that might indicate injection attacks."""
        return self.security_validator.detect_suspicious_patterns(text)
    
    def _log_security_event(self, event_type: str, details: str):
        """Log security events for monitoring."""
        self.security_validator.log_security_event(event_type, details)
    
    def _make_request(self, messages: List[Dict], max_tokens: int = 1000, 
                     temperature: float = 0.1, max_retries: int = 2) -> Optional[str]:
        """Make a request to OpenRouter API with retry logic."""
        return self.api_client.make_request(messages, max_tokens, temperature, max_retries)
    
    def _extract_json_from_response(self, response: str) -> str:
        """Extract JSON from potentially messy LLM response."""
        return self.response_parser.json_parser.extract_json_from_response(response)
    
    def _clean_json_response(self, response: str) -> str:
        """Clean and fix common JSON formatting issues in LLM responses."""
        return self.response_parser.json_parser.clean_json_response(response)
    
    def _parse_json_with_fallbacks(self, text: str, expected_type: str = "any"):
        """Parse JSON with multiple fallback strategies."""
        return self.response_parser.json_parser.parse_json_with_fallbacks(text, expected_type)
    
    # Main API methods - delegate to medical NLP client
    
    def extract_medical_entities(self, text: str) -> List[Dict]:
        """Extract medical entities using OpenRouter LLM."""
        return self.medical_nlp_client.extract_medical_entities(text)
    
    def categorize_soap(self, text: str, entities: List[Dict]) -> Dict[str, List[Dict]]:
        """Categorize entities into SOAP categories."""
        return self.medical_nlp_client.categorize_soap(text, entities)
    
    def extract_relationships(self, text: str, entities: List[Dict]) -> List[Dict]:
        """Extract relationships between medical entities."""
        return self.medical_nlp_client.extract_relationships(text, entities)
    
    def process_clinical_text(self, text: str) -> Dict:
        """Complete pipeline to process clinical text."""
        return self.medical_nlp_client.process_clinical_text(text)
    
    # Utility methods
    
    def check_api_status(self) -> bool:
        """Check if the API is accessible and properly configured."""
        return self.api_client.check_api_status()
    
    def get_client_info(self) -> Dict:
        """Get comprehensive information about the client configuration."""
        return {
            "client_type": "refactored_modular",
            "original_api_compatible": True,
            "components": {
                "medical_nlp_client": "soap_kg.utils.medical_nlp_client.MedicalNLPClient",
                "api_client": "soap_kg.utils.api_client.OpenRouterApiClient", 
                "security_validator": "soap_kg.utils.security.SecurityValidator",
                "response_parser": "soap_kg.utils.json_parser.ResponseParser"
            },
            **self.medical_nlp_client.get_client_info()
        }