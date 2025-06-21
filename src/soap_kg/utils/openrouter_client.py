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
    
    def _clean_json_response(self, response: str) -> str:
        """Clean and fix common JSON formatting issues in LLM responses"""
        # Remove markdown code blocks
        response = response.strip()
        if response.startswith('```json'):
            response = response[7:]
        elif response.startswith('```'):
            response = response[3:]
        
        if response.endswith('```'):
            response = response[:-3]
        
        # Remove any leading/trailing text that's not JSON
        lines = response.split('\n')
        json_lines = []
        in_json = False
        
        for line in lines:
            line = line.strip()
            if line.startswith('[') or line.startswith('{'):
                in_json = True
            if in_json:
                json_lines.append(line)
            if line.endswith(']') or line.endswith('}'):
                break
        
        cleaned = '\n'.join(json_lines).strip()
        
        # Fix common JSON issues
        import re
        # Fix trailing commas
        cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)
        # Fix unescaped quotes in strings (basic attempt)
        cleaned = re.sub(r'(?<!\\)"([^"]*)"([^,:}\]]*)"', r'"\1\2"', cleaned)
        
        return cleaned
    
    def extract_medical_entities(self, text: str) -> List[Dict]:
        """Extract medical entities using OpenRouter LLM"""
        # Return empty list if no API key - let rule-based system handle it
        if not self.api_key:
            logger.info("No OpenRouter API key provided, skipping LLM entity extraction")
            return []
            
        prompt = f"""
        Extract medical entities from the following clinical text. 
        Return a JSON list of entities with the format:
        [{{"text": "entity_text", "type": "DISEASE|SYMPTOM|MEDICATION|PROCEDURE|ANATOMY|LAB_VALUE|VITAL_SIGN|TREATMENT", "confidence": 0.9}}]
        
        Clinical text: {text}
        
        Only return the JSON, no additional text.
        """
        
        messages = [{"role": "user", "content": prompt}]
        response = self._make_request(messages, max_tokens=1500)
        
        if response:
            try:
                # Clean response to extract JSON
                cleaned_response = self._clean_json_response(response)
                if not cleaned_response or cleaned_response.strip() == "":
                    logger.warning("Empty cleaned response from LLM")
                    return []
                    
                entities = json.loads(cleaned_response)
                return entities if isinstance(entities, list) else []
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse entity extraction response: {e}")
                logger.debug(f"Raw response: {response[:200]}...")
                logger.debug(f"Cleaned response: {cleaned_response[:200]}...")
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
        
        Return a JSON object with SOAP categories:
        {{
            "subjective": [entities that represent patient symptoms, complaints, history],
            "objective": [entities that represent vital signs, lab results, physical exam findings],
            "assessment": [entities that represent diagnoses, impressions, evaluations],
            "plan": [entities that represent treatments, medications, procedures, follow-up]
        }}
        
        Only return the JSON, no additional text.
        """
        
        messages = [{"role": "user", "content": prompt}]
        response = self._make_request(messages, max_tokens=2000)
        
        if response:
            try:
                cleaned_response = self._clean_json_response(response)
                if not cleaned_response or cleaned_response.strip() == "":
                    logger.warning("Empty cleaned response from LLM")
                    return {"subjective": [], "objective": [], "assessment": [], "plan": []}
                    
                soap_categories = json.loads(cleaned_response)
                return soap_categories
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse SOAP categorization response: {e}")
                logger.debug(f"Raw response: {response[:200]}...")
                logger.debug(f"Cleaned response: {cleaned_response[:200]}...")
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
        
        Return a JSON list of relationships:
        [{{
            "source": "entity1_text",
            "target": "entity2_text", 
            "relation": "TREATS|CAUSES|INDICATES|MEASURED_BY|LOCATED_IN|HAS_SYMPTOM|PRESCRIBED_FOR|DIAGNOSED_WITH",
            "confidence": 0.9
        }}]
        
        Only return relationships that are explicitly supported by the text.
        Only return the JSON, no additional text.
        """
        
        messages = [{"role": "user", "content": prompt}]
        response = self._make_request(messages, max_tokens=2000)
        
        if response:
            try:
                cleaned_response = self._clean_json_response(response)
                if not cleaned_response or cleaned_response.strip() == "":
                    logger.warning("Empty cleaned response from LLM")
                    return []
                    
                relationships = json.loads(cleaned_response)
                return relationships if isinstance(relationships, list) else []
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse relationship extraction response: {e}")
                logger.debug(f"Raw response: {response[:200]}...")
                logger.debug(f"Cleaned response: {cleaned_response[:200]}...")
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