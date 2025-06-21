import requests
import json
import time
from typing import Dict, List, Optional, Any
from soap_kg.config import Config
import logging

logger = logging.getLogger(__name__)

class OpenRouterClient:
    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or Config.OPENROUTER_API_KEY
        self.base_url = Config.OPENROUTER_BASE_URL
        self.model = model or Config.DEFAULT_MODEL
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/your-repo/soap-kg",
            "X-Title": "SOAP Knowledge Graph Generator"
        }
        
    def _make_request(self, messages: List[Dict], max_tokens: int = 1000, 
                     temperature: float = 0.1, max_retries: int = 2) -> Optional[str]:
        """Make a request to OpenRouter API with retry logic"""
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
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=60  # Increased timeout
                )
                
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
                    import time
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"OpenRouter API error: {response.status_code} - {response.text}")
                    if attempt < max_retries:
                        continue
                    return None
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Request timeout on attempt {attempt + 1}")
                if attempt < max_retries:
                    continue
                return None
            except Exception as e:
                logger.error(f"Error making OpenRouter request on attempt {attempt + 1}: {e}")
                if attempt < max_retries:
                    continue
                return None
        
        return None
    
    def _extract_json_from_response(self, response: str) -> str:
        """Extract JSON from potentially messy LLM response"""
        import re
        
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
    
    def _clean_json_response(self, response: str) -> str:
        """Clean and fix common JSON formatting issues in LLM responses"""
        import re
        
        # First extract the JSON part
        json_text = self._extract_json_from_response(response)
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
    
    def _parse_json_with_fallbacks(self, text: str, expected_type: str = "any") -> any:
        """Parse JSON with multiple fallback strategies"""
        import re
        
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
            cleaned = self._clean_json_response(text)
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
        logger.warning(f"All JSON parsing strategies failed for text: {text[:100]}...")
        if expected_type == "list":
            return []
        elif expected_type == "dict":
            return {}
        return None
    
    def extract_medical_entities(self, text: str) -> List[Dict]:
        """Extract medical entities using OpenRouter LLM"""
        # Return empty list if no API key - let rule-based system handle it
        if not self.api_key:
            logger.info("No OpenRouter API key provided, skipping LLM entity extraction")
            return []
            
        prompt = f"""
        Extract medical entities from the following clinical text. 
        Return ONLY a valid JSON array with this exact format:
        [
          {{"text": "entity_text", "type": "MEDICATION", "confidence": 0.9}},
          {{"text": "entity_text", "type": "DISEASE", "confidence": 0.8}}
        ]
        
        Valid types: DISEASE, SYMPTOM, MEDICATION, PROCEDURE, ANATOMY, LAB_VALUE, VITAL_SIGN, TREATMENT
        
        Clinical text: {text}
        
        IMPORTANT: 
        - Return ONLY the JSON array, no explanation or markdown
        - Use double quotes for all strings
        - Include confidence between 0.0 and 1.0
        - If no entities found, return []
        """
        
        messages = [{"role": "user", "content": prompt}]
        response = self._make_request(messages, max_tokens=1500)
        
        if response:
            try:
                # Use robust parsing with fallbacks
                entities = self._parse_json_with_fallbacks(response, expected_type="list")
                if entities is None:
                    logger.warning("Failed to parse entity extraction response with all strategies")
                    logger.debug(f"Raw response: {response[:300]}...")
                    return []
                return entities if isinstance(entities, list) else []
            except Exception as e:
                logger.error(f"Unexpected error parsing entity extraction response: {e}")
                logger.debug(f"Raw response: {response[:300]}...")
                return []
        
        return []
    
    def categorize_soap(self, text: str, entities: List[Dict]) -> Dict[str, List[Dict]]:
        """Categorize entities into SOAP categories"""
        # Return empty structure if no API key - let rule-based system handle it
        if not self.api_key:
            logger.info("No OpenRouter API key provided, skipping LLM SOAP categorization")
            return {"subjective": [], "objective": [], "assessment": [], "plan": []}
            
        entities_text = json.dumps(entities, indent=2)
        
        prompt = f"""
        Categorize the following medical entities into SOAP categories based on the clinical text context.
        
        Clinical text: {text}
        
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
        response = self._make_request(messages, max_tokens=2000)
        
        if response:
            try:
                # Use robust parsing with fallbacks
                soap_categories = self._parse_json_with_fallbacks(response, expected_type="dict")
                if soap_categories is None:
                    logger.warning("Failed to parse SOAP categorization response with all strategies")
                    logger.debug(f"Raw response: {response[:300]}...")
                    return {"subjective": [], "objective": [], "assessment": [], "plan": []}
                
                # Ensure all expected keys exist and validate structure
                default_categories = {"subjective": [], "objective": [], "assessment": [], "plan": []}
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
                logger.debug(f"Raw response: {response[:300]}...")
                return {"subjective": [], "objective": [], "assessment": [], "plan": []}
        
        return {"subjective": [], "objective": [], "assessment": [], "plan": []}
    
    def extract_relationships(self, text: str, entities: List[Dict]) -> List[Dict]:
        """Extract relationships between medical entities"""
        # Return empty list if no API key - let rule-based system handle it
        if not self.api_key:
            logger.info("No OpenRouter API key provided, skipping LLM relationship extraction")
            return []
            
        entities_text = json.dumps(entities, indent=2)
        
        prompt = f"""
        Extract relationships between medical entities from the clinical text.
        
        Clinical text: {text}
        
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
        response = self._make_request(messages, max_tokens=2000)
        
        if response:
            try:
                # Use robust parsing with fallbacks
                relationships = self._parse_json_with_fallbacks(response, expected_type="list")
                if relationships is None:
                    logger.warning("Failed to parse relationship extraction response with all strategies")
                    logger.debug(f"Raw response: {response[:300]}...")
                    return []
                return relationships if isinstance(relationships, list) else []
            except Exception as e:
                logger.error(f"Unexpected error parsing relationship extraction response: {e}")
                logger.debug(f"Raw response: {response[:300]}...")
                return []
        
        return []
    
    def process_clinical_text(self, text: str) -> Dict:
        """Complete pipeline to process clinical text"""
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