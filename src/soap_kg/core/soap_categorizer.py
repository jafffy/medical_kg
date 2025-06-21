from typing import List, Dict, Optional
import logging
from soap_kg.utils.openrouter_client import OpenRouterClient
from soap_kg.models.soap_schema import MedicalEntity, SOAPCategory, EntityType
import re

logger = logging.getLogger(__name__)

class SOAPCategorizer:
    def __init__(self, openrouter_client: OpenRouterClient = None):
        self.client = openrouter_client or OpenRouterClient()
        
        # Rule-based SOAP categorization patterns
        self.soap_patterns = {
            SOAPCategory.SUBJECTIVE: [
                r'\b(?:patient states?|patient reports?|patient complains?|patient describes?)\b',
                r'\b(?:chief complaint|cc|history of present illness|hpi)\b',
                r'\b(?:patient denies|patient admits|patient feels)\b',
                r'\b(?:pain|discomfort|nausea|fatigue|weakness|dizziness)\b',
                r'\b(?:family history|social history|surgical history)\b'
            ],
            SOAPCategory.OBJECTIVE: [
                r'\b(?:vital signs|vs|temperature|bp|blood pressure|heart rate|hr)\b',
                r'\b(?:physical exam|examination|inspection|palpation|auscultation)\b',
                r'\b(?:lab results?|laboratory|cbc|bmp|liver function|cardiac enzymes)\b',
                r'\b(?:imaging|x-ray|ct|mri|ultrasound|ecg|ekg)\b',
                r'\b(?:\d+/\d+\s*mmhg|\d+\s*bpm|\d+\s*mg/dl|\d+\s*mcg)\b'
            ],
            SOAPCategory.ASSESSMENT: [
                r'\b(?:diagnosis|dx|impression|assessment|differential)\b',
                r'\b(?:likely|probable|possible|rule out|r/o|working diagnosis)\b',
                r'\b(?:acute|chronic|stable|unstable|improving|worsening)\b',
                r'\b(?:icd|diagnostic code|primary diagnosis|secondary diagnosis)\b'
            ],
            SOAPCategory.PLAN: [
                r'\b(?:plan|treatment|therapy|management|intervention)\b',
                r'\b(?:medication|prescription|rx|drug|dosage)\b',
                r'\b(?:surgery|procedure|operation|follow-up|discharge)\b',
                r'\b(?:monitor|observe|continue|discontinue|start|stop)\b',
                r'\b(?:patient education|counseling|lifestyle modification)\b'
            ]
        }
        
        # Entity type to SOAP category mapping (default assignments)
        self.entity_soap_mapping = {
            EntityType.SYMPTOM: SOAPCategory.SUBJECTIVE,
            EntityType.VITAL_SIGN: SOAPCategory.OBJECTIVE,
            EntityType.LAB_VALUE: SOAPCategory.OBJECTIVE,
            EntityType.DISEASE: SOAPCategory.ASSESSMENT,
            EntityType.MEDICATION: SOAPCategory.PLAN,
            EntityType.TREATMENT: SOAPCategory.PLAN,
            EntityType.PROCEDURE: SOAPCategory.PLAN,
            EntityType.ANATOMY: SOAPCategory.OBJECTIVE
        }
    
    def categorize_text_patterns(self, text: str) -> Dict[SOAPCategory, float]:
        """Categorize text into SOAP categories using pattern matching"""
        scores = {category: 0.0 for category in SOAPCategory}
        
        text_lower = text.lower()
        
        for category, patterns in self.soap_patterns.items():
            category_score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower, re.IGNORECASE))
                category_score += matches
            
            # Normalize by number of patterns and text length
            if len(patterns) > 0 and len(text) > 0:
                scores[category] = category_score / (len(patterns) * (len(text) / 100))
        
        return scores
    
    def categorize_entity_by_type(self, entity: MedicalEntity) -> SOAPCategory:
        """Categorize entity based on its type"""
        return self.entity_soap_mapping.get(entity.entity_type, SOAPCategory.OBJECTIVE)
    
    def categorize_with_llm(self, text: str, entities: List[MedicalEntity]) -> Dict[str, SOAPCategory]:
        """Categorize entities using OpenRouter LLM"""
        try:
            entity_dicts = [
                {"text": e.text, "type": e.entity_type.value, "id": e.id}
                for e in entities
            ]
            
            soap_result = self.client.categorize_soap(text, entity_dicts)
            
            # Convert result to entity ID -> SOAPCategory mapping
            entity_categories = {}
            
            for category_name, entity_list in soap_result.items():
                try:
                    soap_category = SOAPCategory(category_name.lower())
                    for entity_dict in entity_list:
                        # Find matching entity by text
                        for entity in entities:
                            if entity.text.lower() == entity_dict.get('text', '').lower():
                                entity_categories[entity.id] = soap_category
                                break
                except ValueError:
                    continue
            
            return entity_categories
            
        except Exception as e:
            logger.error(f"LLM SOAP categorization failed: {e}")
            return {}
    
    def categorize_by_context(self, text: str, entities: List[MedicalEntity]) -> Dict[str, SOAPCategory]:
        """Categorize entities based on their context in the text"""
        entity_categories = {}
        
        # Split text into sentences for better context analysis
        sentences = re.split(r'[.!?]+', text)
        
        for entity in entities:
            # Find sentences containing this entity
            entity_sentences = []
            for sentence in sentences:
                if entity.text.lower() in sentence.lower():
                    entity_sentences.append(sentence)
            
            if not entity_sentences:
                # Default categorization based on entity type
                entity_categories[entity.id] = self.categorize_entity_by_type(entity)
                continue
            
            # Analyze context of sentences containing the entity
            context_scores = {category: 0.0 for category in SOAPCategory}
            
            for sentence in entity_sentences:
                sentence_scores = self.categorize_text_patterns(sentence)
                for category, score in sentence_scores.items():
                    context_scores[category] += score
            
            # Choose category with highest score
            best_category = max(context_scores, key=context_scores.get)
            
            # If no clear pattern match, use entity type default
            if context_scores[best_category] == 0:
                best_category = self.categorize_entity_by_type(entity)
            
            entity_categories[entity.id] = best_category
        
        return entity_categories
    
    def categorize_entities(self, text: str, entities: List[MedicalEntity], 
                          use_llm: bool = True) -> List[MedicalEntity]:
        """Main entity categorization pipeline"""
        logger.info(f"Categorizing {len(entities)} entities into SOAP categories")
        
        categorization_results = []
        
        # Try LLM categorization first
        if use_llm:
            llm_categories = self.categorize_with_llm(text, entities)
            categorization_results.append(llm_categories)
        
        # Context-based categorization
        context_categories = self.categorize_by_context(text, entities)
        categorization_results.append(context_categories)
        
        # Entity type-based categorization (fallback)
        type_categories = {
            entity.id: self.categorize_entity_by_type(entity) 
            for entity in entities
        }
        categorization_results.append(type_categories)
        
        # Combine results with priority (LLM > Context > Type)
        final_categories = {}
        for entity in entities:
            entity_id = entity.id
            
            # Use first available categorization result
            for category_dict in categorization_results:
                if entity_id in category_dict:
                    final_categories[entity_id] = category_dict[entity_id]
                    break
        
        # Update entities with SOAP categories
        updated_entities = []
        for entity in entities:
            entity.soap_category = final_categories.get(entity.id, SOAPCategory.OBJECTIVE)
            updated_entities.append(entity)
        
        # Log categorization statistics
        category_counts = {}
        for entity in updated_entities:
            category = entity.soap_category
            category_counts[category] = category_counts.get(category, 0) + 1
        
        logger.info(f"SOAP categorization results: {category_counts}")
        
        return updated_entities
    
    def create_soap_structure(self, entities: List[MedicalEntity]) -> Dict[SOAPCategory, List[MedicalEntity]]:
        """Organize entities into SOAP structure"""
        soap_structure = {category: [] for category in SOAPCategory}
        
        for entity in entities:
            soap_structure[entity.soap_category].append(entity)
        
        return soap_structure
    
    def validate_soap_categorization(self, entities: List[MedicalEntity]) -> Dict[str, any]:
        """Validate and provide insights on SOAP categorization"""
        validation_results = {
            "total_entities": len(entities),
            "category_distribution": {},
            "potential_issues": [],
            "confidence_stats": {}
        }
        
        # Category distribution
        for category in SOAPCategory:
            category_entities = [e for e in entities if e.soap_category == category]
            validation_results["category_distribution"][category.value] = len(category_entities)
            
            # Confidence statistics per category
            if category_entities:
                confidences = [e.confidence for e in category_entities]
                validation_results["confidence_stats"][category.value] = {
                    "mean": sum(confidences) / len(confidences),
                    "min": min(confidences),
                    "max": max(confidences)
                }
        
        # Check for potential issues
        subjective_count = validation_results["category_distribution"].get("subjective", 0)
        objective_count = validation_results["category_distribution"].get("objective", 0)
        assessment_count = validation_results["category_distribution"].get("assessment", 0)
        plan_count = validation_results["category_distribution"].get("plan", 0)
        
        if subjective_count == 0:
            validation_results["potential_issues"].append("No subjective entities found - missing patient symptoms/complaints")
        
        if assessment_count == 0:
            validation_results["potential_issues"].append("No assessment entities found - missing diagnoses/impressions")
        
        if plan_count == 0:
            validation_results["potential_issues"].append("No plan entities found - missing treatments/interventions")
        
        if objective_count > (subjective_count + assessment_count + plan_count):
            validation_results["potential_issues"].append("Disproportionately high objective entities - check categorization")
        
        return validation_results