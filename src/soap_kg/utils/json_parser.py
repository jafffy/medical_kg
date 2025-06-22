"""
JSON parsing utilities for handling LLM responses.

This module provides robust JSON parsing capabilities with multiple fallback strategies
to handle the often messy JSON output from language models.
"""

import json
import re
import logging
from typing import Any, Union
from soap_kg.utils.security import SecurityValidator

logger = logging.getLogger(__name__)


class LLMJsonParser:
    """Parser for handling JSON responses from LLM APIs."""
    
    def __init__(self):
        self.security_validator = SecurityValidator()
    
    def extract_json_from_response(self, response: str) -> str:
        """Extract JSON from potentially messy LLM response."""
        # Remove markdown code blocks
        response = response.strip()
        if response.startswith('```json'):
            response = response[7:]
        elif response.startswith('```'):
            response = response[3:]
        
        if response.endswith('```'):
            response = response[:-3]
        
        # Try to find JSON structures using regex
        # Look for JSON arrays or objects
        json_patterns = [
            r'\[.*?\]',  # Array pattern
            r'\{.*?\}',  # Object pattern
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            if matches:
                # Return the longest match (likely the most complete)
                return max(matches, key=len)
        
        # Fallback: try line-by-line extraction
        lines = response.split('\n')
        json_lines = []
        bracket_count = 0
        brace_count = 0
        in_json = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Start of JSON
            if line.startswith('[') or line.startswith('{'):
                in_json = True
                json_lines = [line]  # Reset and start fresh
                bracket_count = line.count('[') - line.count(']')
                brace_count = line.count('{') - line.count('}')
                continue
            
            if in_json:
                json_lines.append(line)
                bracket_count += line.count('[') - line.count(']')
                brace_count += line.count('{') - line.count('}')
                
                # Check if we've closed all brackets/braces
                if bracket_count <= 0 and brace_count <= 0:
                    break
        
        return '\n'.join(json_lines).strip()
    
    def clean_json_response(self, response: str) -> str:
        """Clean and fix common JSON formatting issues in LLM responses."""
        # First extract the JSON part
        json_text = self.extract_json_from_response(response)
        if not json_text:
            return ""
        
        # Clean up common issues
        cleaned = json_text.strip()
        
        # Remove comments (// or /* */)
        cleaned = re.sub(r'//.*?\n', '\n', cleaned)
        cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)
        
        # Fix trailing commas
        cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)
        
        # Fix single quotes to double quotes (but be careful with apostrophes)
        cleaned = re.sub(r"(?<!\\)'([^']*)'(?=\s*[,:\]}])", r'"\1"', cleaned)
        
        # Fix unescaped quotes in values
        # This is a simplified approach - proper fix would need full parsing
        lines = cleaned.split('\n')
        fixed_lines = []
        
        for line in lines:
            # Skip if this line looks like a key-value pair that's already properly quoted
            if ':' in line and line.strip().startswith('"') and line.count('"') >= 2:
                # Try to fix internal quotes in the value part
                if line.count('"') > 2:  # More than just key and start of value
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        key_part = parts[0]
                        value_part = parts[1].strip()
                        if value_part.startswith('"') and value_part.count('"') > 1:
                            # Extract the value and escape internal quotes
                            end_quote = value_part.rfind('"')
                            if end_quote > 0:
                                value_content = value_part[1:end_quote]
                                value_suffix = value_part[end_quote+1:]
                                # Escape internal quotes
                                value_content = value_content.replace('"', '\\"')
                                line = f'{key_part}: "{value_content}"{value_suffix}'
            fixed_lines.append(line)
        
        cleaned = '\n'.join(fixed_lines)
        
        # Final cleanup
        cleaned = re.sub(r'\s+', ' ', cleaned)  # Normalize whitespace
        cleaned = cleaned.replace('\\n', ' ').replace('\\t', ' ')  # Remove literal newlines/tabs
        
        return cleaned
    
    def parse_json_with_fallbacks(self, text: str, expected_type: str = "any") -> Any:
        """Parse JSON with multiple fallback strategies."""
        if not text or not text.strip():
            if expected_type == "list":
                return []
            elif expected_type == "dict":
                return {}
            return None
        
        # Strategy 1: Direct parsing
        try:
            result = json.loads(text)
            return result
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Clean and parse
        try:
            cleaned = self.clean_json_response(text)
            if cleaned:
                result = json.loads(cleaned)
                return result
        except json.JSONDecodeError:
            pass
        
        # Strategy 3: Extract and parse individual JSON objects/arrays
        if expected_type == "list":
            # Try to extract individual array elements
            items = []
            # Look for individual objects within the text
            object_pattern = r'\{[^{}]*\}'
            matches = re.findall(object_pattern, text)
            for match in matches:
                try:
                    item = json.loads(match)
                    items.append(item)
                except json.JSONDecodeError:
                    continue
            if items:
                return items
        
        # Strategy 4: Manual parsing for simple cases
        if expected_type == "list" and '[' in text and ']' in text:
            # Try to extract content between [ and ]
            start = text.find('[')
            end = text.rfind(']')
            if start != -1 and end != -1 and end > start:
                array_content = text[start:end+1]
                try:
                    return json.loads(array_content)
                except json.JSONDecodeError:
                    pass
        
        # Strategy 5: Return appropriate empty structure
        safe_text = self.security_validator.mask_sensitive_data(text[:100])
        logger.warning(f"All JSON parsing strategies failed for text: {safe_text}...")
        if expected_type == "list":
            return []
        elif expected_type == "dict":
            return {}
        return None


class ResponseParser:
    """High-level response parsing interface."""
    
    def __init__(self):
        self.json_parser = LLMJsonParser()
        self.security_validator = SecurityValidator()
    
    def parse_entity_response(self, response: str) -> list:
        """Parse entity extraction response."""
        try:
            entities = self.json_parser.parse_json_with_fallbacks(response, expected_type="list")
            if entities is None:
                logger.warning("Failed to parse entity extraction response with all strategies")
                safe_response = self.security_validator.mask_sensitive_data(response[:300])
                logger.debug(f"Raw response: {safe_response}...")
                return []
            return entities if isinstance(entities, list) else []
        except Exception as e:
            logger.error(f"Unexpected error parsing entity extraction response: {e}")
            safe_response = self.security_validator.mask_sensitive_data(response[:300])
            logger.debug(f"Raw response: {safe_response}...")
            return []
    
    def parse_soap_response(self, response: str) -> dict:
        """Parse SOAP categorization response."""
        default_categories = {"subjective": [], "objective": [], "assessment": [], "plan": []}
        
        try:
            soap_categories = self.json_parser.parse_json_with_fallbacks(response, expected_type="dict")
            if soap_categories is None:
                logger.warning("Failed to parse SOAP categorization response with all strategies")
                safe_response = self.security_validator.mask_sensitive_data(response[:300])
                logger.debug(f"Raw response: {safe_response}...")
                return default_categories
            
            # Ensure all expected keys exist and validate structure
            if isinstance(soap_categories, dict):
                # Ensure all required keys are present
                for key in default_categories:
                    if key not in soap_categories:
                        soap_categories[key] = []
                    # Ensure each value is a list
                    elif not isinstance(soap_categories[key], list):
                        logger.warning(f"SOAP category '{key}' is not a list, converting: {type(soap_categories[key])}")
                        if isinstance(soap_categories[key], str):
                            # If it's a string, try to treat it as a single entity
                            soap_categories[key] = [soap_categories[key]] if soap_categories[key].strip() else []
                        else:
                            soap_categories[key] = []
                
                # Validate each list contains proper items
                for key, items in soap_categories.items():
                    if isinstance(items, list):
                        validated_items = []
                        for item in items:
                            if isinstance(item, dict):
                                validated_items.append(item)
                            elif isinstance(item, str) and item.strip():
                                # Convert string to simple dict format
                                validated_items.append({"text": item.strip()})
                            # Skip invalid items
                        soap_categories[key] = validated_items
                
                return soap_categories
            else:
                logger.warning(f"SOAP categorization returned non-dict: {type(soap_categories)}")
                return default_categories
        except Exception as e:
            logger.error(f"Unexpected error parsing SOAP categorization response: {e}")
            safe_response = self.security_validator.mask_sensitive_data(response[:300])
            logger.debug(f"Raw response: {safe_response}...")
            return default_categories
        
        return default_categories
    
    def parse_relationship_response(self, response: str) -> list:
        """Parse relationship extraction response."""
        try:
            relationships = self.json_parser.parse_json_with_fallbacks(response, expected_type="list")
            if relationships is None:
                logger.warning("Failed to parse relationship extraction response with all strategies")
                safe_response = self.security_validator.mask_sensitive_data(response[:300])
                logger.debug(f"Raw response: {safe_response}...")
                return []
            return relationships if isinstance(relationships, list) else []
        except Exception as e:
            logger.error(f"Unexpected error parsing relationship extraction response: {e}")
            safe_response = self.security_validator.mask_sensitive_data(response[:300])
            logger.debug(f"Raw response: {safe_response}...")
            return []