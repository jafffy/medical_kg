from typing import List, Dict, Optional, Tuple
import logging
from openrouter_client import OpenRouterClient
from text_preprocessor import MedicalTextPreprocessor
from soap_schema import MedicalEntity, EntityType, SOAPCategory
import uuid

logger = logging.getLogger(__name__)

class MedicalNER:
    def __init__(self, openrouter_client: OpenRouterClient = None):
        self.client = openrouter_client or OpenRouterClient()
        self.preprocessor = MedicalTextPreprocessor()
        
        # Medical entity patterns for fallback rule-based NER
        self.entity_patterns = {
            EntityType.DISEASE: [
                r'\b(?:hypertension|diabetes|pneumonia|sepsis|copd|asthma|cancer|tumor|infection|fever)\b',
                r'\b(?:mi|myocardial infarction|stroke|heart failure|cardiac arrest)\b',
                r'\b(?:depression|anxiety|dementia|delirium|psychosis)\b'
            ],
            EntityType.MEDICATION: [
                r'\b(?:aspirin|metformin|insulin|lisinopril|atorvastatin|amlodipine)\b',
                r'\b(?:morphine|fentanyl|propofol|midazolam|lorazepam)\b',
                r'\b(?:antibiotics?|antibiotic|penicillin|amoxicillin|vancomycin)\b'
            ],
            EntityType.PROCEDURE: [
                r'\b(?:surgery|operation|biopsy|intubation|catheterization)\b',
                r'\b(?:ecg|ekg|x-ray|ct scan|mri|ultrasound|echo)\b',
                r'\b(?:blood transfusion|dialysis|chemotherapy|radiation)\b'
            ],
            EntityType.ANATOMY: [
                r'\b(?:heart|lung|liver|kidney|brain|stomach|chest|abdomen)\b',
                r'\b(?:left ventricle|right atrium|aorta|pulmonary artery)\b',
                r'\b(?:head|neck|arm|leg|hand|foot|back|spine)\b'
            ],
            EntityType.VITAL_SIGN: [
                r'\b(?:blood pressure|heart rate|temperature|respiratory rate)\b',
                r'\b(?:bp|hr|temp|rr|pulse|o2 sat|oxygen saturation)\b',
                r'\b(?:\d+/\d+\s*mmhg|\d+\s*bpm|\d+\s*rpm)\b'
            ],
            EntityType.LAB_VALUE: [
                r'\b(?:glucose|creatinine|bun|hemoglobin|hematocrit|wbc|rbc)\b',
                r'\b(?:sodium|potassium|chloride|co2|anion gap)\b',
                r'\b(?:troponin|bnp|d-dimer|lactate|procalcitonin)\b'
            ]
        }
    
    def extract_entities_llm(self, text: str) -> List[Dict]:
        """Extract entities using OpenRouter LLM"""
        try:
            entities = self.client.extract_medical_entities(text)
            return entities
        except Exception as e:
            logger.error(f"LLM entity extraction failed: {e}")
            return []
    
    def extract_entities_rules(self, text: str) -> List[Dict]:
        """Fallback rule-based entity extraction"""
        import re
        entities = []
        
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entities.append({
                        'text': match.group().strip(),
                        'type': entity_type.value.upper(),
                        'confidence': 0.7,
                        'start': match.start(),
                        'end': match.end()
                    })
        
        return entities
    
    def extract_structured_entities(self, preprocessed_data: Dict) -> List[Dict]:
        """Extract entities from preprocessed structured data"""
        entities = []
        
        # Extract vital signs as entities
        for vital in preprocessed_data.get('vital_signs', []):
            entities.append({
                'text': vital['text'],
                'type': 'VITAL_SIGN',
                'confidence': 0.9,
                'value': vital['value'],
                'vital_type': vital['type']
            })
        
        # Extract medications as entities
        for med in preprocessed_data.get('medications', []):
            entities.append({
                'text': med['full_text'],
                'type': 'MEDICATION',
                'confidence': 0.8,
                'medication_name': med['medication'],
                'dose': med.get('dose'),
                'unit': med.get('unit')
            })
        
        return entities
    
    def merge_and_deduplicate_entities(self, entity_lists: List[List[Dict]]) -> List[Dict]:
        """Merge entities from different sources and remove duplicates"""
        all_entities = []
        for entity_list in entity_lists:
            all_entities.extend(entity_list)
        
        # Simple deduplication based on text similarity
        unique_entities = []
        seen_texts = set()
        
        for entity in all_entities:
            text_lower = entity['text'].lower().strip()
            if text_lower not in seen_texts:
                seen_texts.add(text_lower)
                unique_entities.append(entity)
            else:
                # If we've seen this text, keep the one with higher confidence
                for i, existing in enumerate(unique_entities):
                    if existing['text'].lower().strip() == text_lower:
                        if entity.get('confidence', 0) > existing.get('confidence', 0):
                            unique_entities[i] = entity
                        break
        
        return unique_entities
    
    def convert_to_medical_entities(self, entities: List[Dict], 
                                  soap_category: SOAPCategory = SOAPCategory.OBJECTIVE) -> List[MedicalEntity]:
        """Convert raw entities to MedicalEntity objects"""
        medical_entities = []
        
        for entity in entities:
            try:
                # Map string entity types to EntityType enum
                entity_type_str = entity.get('type', 'TREATMENT').upper()
                try:
                    entity_type = EntityType(entity_type_str.lower())
                except ValueError:
                    # Default to TREATMENT if type not recognized
                    entity_type = EntityType.TREATMENT
                
                medical_entity = MedicalEntity(
                    id=str(uuid.uuid4()),
                    text=entity['text'],
                    entity_type=entity_type,
                    soap_category=soap_category,
                    confidence=entity.get('confidence', 0.5),
                    metadata={
                        'extraction_method': 'llm' if 'confidence' in entity and entity['confidence'] > 0.8 else 'rules',
                        **{k: v for k, v in entity.items() if k not in ['text', 'type', 'confidence']}
                    }
                )
                
                medical_entities.append(medical_entity)
            except Exception as e:
                logger.error(f"Error converting entity {entity}: {e}")
                continue
        
        return medical_entities
    
    def extract_entities(self, text: str, use_llm: bool = True) -> List[MedicalEntity]:
        """Main entity extraction pipeline"""
        logger.info(f"Extracting entities from text: {text[:100]}...")
        
        # Preprocess text
        preprocessed = self.preprocessor.preprocess_clinical_text(text)
        cleaned_text = preprocessed['cleaned_text']
        
        entity_sources = []
        
        # Extract from structured data first
        structured_entities = self.extract_structured_entities(preprocessed)
        entity_sources.append(structured_entities)
        
        # Try LLM extraction if available
        if use_llm:
            llm_entities = self.extract_entities_llm(cleaned_text)
            if llm_entities:
                entity_sources.append(llm_entities)
        
        # Always run rule-based as fallback
        rule_entities = self.extract_entities_rules(cleaned_text)
        entity_sources.append(rule_entities)
        
        # Merge and deduplicate
        merged_entities = self.merge_and_deduplicate_entities(entity_sources)
        
        # Convert to MedicalEntity objects
        medical_entities = self.convert_to_medical_entities(merged_entities)
        
        logger.info(f"Extracted {len(medical_entities)} unique entities")
        return medical_entities
    
    def process_patient_texts(self, patient_data: Dict) -> List[MedicalEntity]:
        """Process all clinical texts for a patient"""
        all_entities = []
        
        # Process prescriptions
        for prescription in patient_data.get('prescriptions', []):
            if 'drug' in prescription and prescription['drug']:
                entities = self.extract_entities(str(prescription['drug']))
                all_entities.extend(entities)
        
        # Process diagnosis codes (convert to text first)
        for diagnosis in patient_data.get('diagnoses', []):
            if 'icd_code' in diagnosis:
                # This would need ICD code lookup - simplified for now
                entities = self.extract_entities(f"diagnosis code {diagnosis['icd_code']}")
                all_entities.extend(entities)
        
        return all_entities