"""
Medical NLP client for extracting entities, relationships, and SOAP categorization.

This module provides high-level medical NLP functionality using the OpenRouter API,
with proper input validation and response parsing.
"""

import json
import logging
from typing import Dict, List
from soap_kg.utils.api_client import OpenRouterApiClient
from soap_kg.utils.security import SecurityValidator
from soap_kg.utils.json_parser import ResponseParser

logger = logging.getLogger(__name__)


class MedicalNLPClient:
    """High-level medical NLP interface using OpenRouter API."""
    
    def __init__(self, api_key: str = None, model: str = None):
        self.api_client = OpenRouterApiClient(api_key=api_key, model=model)
        self.security_validator = SecurityValidator()
        self.response_parser = ResponseParser()
    
    def extract_medical_entities(self, text: str) -> List[Dict]:
        """Extract medical entities using OpenRouter LLM."""
        # Return empty list if no API key - let rule-based system handle it
        if not self.api_client.api_key:
            logger.info("No OpenRouter API key provided, skipping LLM entity extraction")
            return []
        
        # Sanitize and validate input
        if not text or not text.strip():
            return []
            
        # Check for suspicious patterns
        if self.security_validator.detect_suspicious_patterns(text):
            logger.warning("Suspicious patterns detected in input text, blocking request")
            return []
            
        # Sanitize input text
        sanitized_text = self.security_validator.sanitize_input_text(text)
            
        prompt = f"""
        Extract medical entities from the following clinical text. 
        Return ONLY a valid JSON array with this exact format:
        [
          {{"text": "entity_text", "type": "MEDICATION", "confidence": 0.9}},
          {{"text": "entity_text", "type": "DISEASE", "confidence": 0.8}}
        ]
        
        Valid types: DISEASE, SYMPTOM, MEDICATION, PROCEDURE, ANATOMY, LAB_VALUE, VITAL_SIGN, TREATMENT
        
        Clinical text: {sanitized_text}
        
        IMPORTANT: 
        - Return ONLY the JSON array, no explanation or markdown
        - Use double quotes for all strings
        - Include confidence between 0.0 and 1.0
        - If no entities found, return []
        """
        
        messages = [{"role": "user", "content": prompt}]
        response = self.api_client.make_request(messages, max_tokens=1500)
        
        if response:
            return self.response_parser.parse_entity_response(response)
        
        return []
    
    def categorize_soap(self, text: str, entities: List[Dict]) -> Dict[str, List[Dict]]:
        """Categorize entities into SOAP categories."""
        # Return empty structure if no API key - let rule-based system handle it
        if not self.api_client.api_key:
            logger.info("No OpenRouter API key provided, skipping LLM SOAP categorization")
            return {"subjective": [], "objective": [], "assessment": [], "plan": []}
        
        # Validate inputs
        if not text or not text.strip():
            return {"subjective": [], "objective": [], "assessment": [], "plan": []}
            
        # Check for suspicious patterns
        if self.security_validator.detect_suspicious_patterns(text):
            logger.warning("Suspicious patterns detected in input text, blocking request")
            return {"subjective": [], "objective": [], "assessment": [], "plan": []}
            
        # Sanitize input text
        sanitized_text = self.security_validator.sanitize_input_text(text)
        
        # Validate entities input
        if not isinstance(entities, list):
            logger.warning("Invalid entities input type, expected list")
            entities = []
            
        entities_text = json.dumps(entities, indent=2)
        
        prompt = f"""
        Categorize the following medical entities into SOAP categories based on the clinical text context.
        
        Clinical text: {sanitized_text}
        
        Entities: {entities_text}
        
        Return ONLY a valid JSON object with this exact format:
        {{
            "subjective": [],
            "objective": [],
            "assessment": [],
            "plan": []
        }}
        
        SOAP definitions:
        - subjective: Patient symptoms, complaints, history, what patient says
        - objective: Vital signs, lab results, physical exam findings, measurable data
        - assessment: Diagnoses, impressions, evaluations, clinical judgment
        - plan: Treatments, medications, procedures, follow-up actions
        
        IMPORTANT: 
        - Return ONLY the JSON object, no explanation or markdown
        - Use double quotes for all strings
        - Each array should contain the relevant entities from the input
        - All four keys must be present even if arrays are empty
        """
        
        messages = [{"role": "user", "content": prompt}]
        response = self.api_client.make_request(messages, max_tokens=2000)
        
        if response:
            return self.response_parser.parse_soap_response(response)
        
        return {"subjective": [], "objective": [], "assessment": [], "plan": []}
    
    def extract_relationships(self, text: str, entities: List[Dict]) -> List[Dict]:
        """Extract relationships between medical entities."""
        # Return empty list if no API key - let rule-based system handle it
        if not self.api_client.api_key:
            logger.info("No OpenRouter API key provided, skipping LLM relationship extraction")
            return []
        
        # Validate inputs
        if not text or not text.strip():
            return []
            
        # Check for suspicious patterns
        if self.security_validator.detect_suspicious_patterns(text):
            logger.warning("Suspicious patterns detected in input text, blocking request")
            return []
            
        # Sanitize input text
        sanitized_text = self.security_validator.sanitize_input_text(text)
        
        # Validate entities input
        if not isinstance(entities, list):
            logger.warning("Invalid entities input type, expected list")
            entities = []
            
        entities_text = json.dumps(entities, indent=2)
        
        prompt = f"""
        Extract relationships between medical entities from the clinical text.
        
        Clinical text: {sanitized_text}
        
        Entities: {entities_text}
        
        Return ONLY a valid JSON array with this exact format:
        [
          {{
            "source": "entity1_text",
            "target": "entity2_text", 
            "relation": "TREATS",
            "confidence": 0.9
          }}
        ]
        
        Valid relations: TREATS, CAUSES, INDICATES, MEASURED_BY, LOCATED_IN, HAS_SYMPTOM, PRESCRIBED_FOR, DIAGNOSED_WITH
        
        IMPORTANT: 
        - Return ONLY the JSON array, no explanation or markdown
        - Use double quotes for all strings
        - Only include relationships explicitly supported by the text
        - Include confidence between 0.0 and 1.0
        - If no relationships found, return []
        """
        
        messages = [{"role": "user", "content": prompt}]
        response = self.api_client.make_request(messages, max_tokens=2000)
        
        if response:
            return self.response_parser.parse_relationship_response(response)
        
        return []
    
    def process_clinical_text(self, text: str) -> Dict:
        """Complete pipeline to process clinical text."""
        logger.info(f"Processing clinical text: {text[:100]}...")
        
        # Extract entities
        entities = self.extract_medical_entities(text)
        logger.info(f"Extracted {len(entities)} entities")
        
        # Categorize into SOAP
        soap_categories = self.categorize_soap(text, entities)
        logger.info("Categorized entities into SOAP")
        
        # Extract relationships
        relationships = self.extract_relationships(text, entities)
        logger.info(f"Extracted {len(relationships)} relationships")
        
        return {
            "text": text,
            "entities": entities,
            "soap_categories": soap_categories,
            "relationships": relationships
        }
    
    def get_client_info(self) -> Dict:
        """Get information about the client configuration."""
        return {
            "nlp_client": {
                "has_api_key": bool(self.api_client.api_key),
                "model": self.api_client.model,
                "base_url": self.api_client.base_url
            },
            "api_client": self.api_client.get_model_info()
        }