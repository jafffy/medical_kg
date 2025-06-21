from typing import List, Dict, Optional, Tuple
import logging
from openrouter_client import OpenRouterClient
from soap_schema import MedicalEntity, MedicalRelation, RelationType, SOAPCategory
import uuid
import re

logger = logging.getLogger(__name__)

class RelationshipExtractor:
    def __init__(self, openrouter_client: OpenRouterClient = None):
        self.client = openrouter_client or OpenRouterClient()
        
        # Rule-based relationship patterns
        self.relation_patterns = {
            RelationType.TREATS: [
                r'({entity1})\s+(?:treats|treating|treatment for|therapy for)\s+({entity2})',
                r'({entity1})\s+(?:given for|prescribed for|administered for)\s+({entity2})',
                r'({entity2})\s+(?:treated with|managed with|given)\s+({entity1})'
            ],
            RelationType.CAUSES: [
                r'({entity1})\s+(?:causes|caused|leads to|results in)\s+({entity2})',
                r'({entity2})\s+(?:due to|caused by|secondary to)\s+({entity1})'
            ],
            RelationType.INDICATES: [
                r'({entity1})\s+(?:indicates|suggests|shows)\s+({entity2})',
                r'({entity2})\s+(?:indicated by|suggested by)\s+({entity1})'
            ],
            RelationType.HAS_SYMPTOM: [
                r'({entity1})\s+(?:presents with|has|experiences|complains of)\s+({entity2})',
                r'({entity2})\s+(?:in|present in)\s+({entity1})'
            ],
            RelationType.DIAGNOSED_WITH: [
                r'({entity1})\s+(?:diagnosed with|has diagnosis of)\s+({entity2})',
                r'({entity2})\s+(?:diagnosed in|found in)\s+({entity1})'
            ],
            RelationType.LOCATED_IN: [
                r'({entity1})\s+(?:in|located in|found in)\s+({entity2})',
                r'({entity2})\s+(?:contains|has)\s+({entity1})'
            ]
        }
    
    def extract_relationships_llm(self, text: str, entities: List[MedicalEntity]) -> List[Dict]:
        """Extract relationships using OpenRouter LLM"""
        try:
            entity_dicts = [
                {"text": e.text, "type": e.entity_type.value, "id": e.id}
                for e in entities
            ]
            relationships = self.client.extract_relationships(text, entity_dicts)
            return relationships
        except Exception as e:
            logger.error(f"LLM relationship extraction failed: {e}")
            return []
    
    def extract_relationships_rules(self, text: str, entities: List[MedicalEntity]) -> List[Dict]:
        """Rule-based relationship extraction"""
        relationships = []
        
        # Create entity lookup by text
        entity_by_text = {e.text.lower(): e for e in entities}
        entity_texts = [e.text for e in entities]
        
        for relation_type, patterns in self.relation_patterns.items():
            for pattern_template in patterns:
                # Try all entity pairs
                for i, entity1 in enumerate(entity_texts):
                    for j, entity2 in enumerate(entity_texts):
                        if i != j:  # Don't relate entity to itself
                            # Create pattern with specific entities
                            pattern = pattern_template.format(
                                entity1=re.escape(entity1),
                                entity2=re.escape(entity2)
                            )
                            
                            matches = re.search(pattern, text, re.IGNORECASE)
                            if matches:
                                relationships.append({
                                    'source': entity1,
                                    'target': entity2,
                                    'relation': relation_type.value.upper(),
                                    'confidence': 0.7,
                                    'pattern_match': matches.group(0)
                                })
        
        return relationships
    
    def extract_cooccurrence_relationships(self, entities: List[MedicalEntity], 
                                         window_size: int = 50) -> List[Dict]:
        """Extract relationships based on entity co-occurrence"""
        relationships = []
        
        # Group entities by SOAP category for better relationship inference
        soap_groups = {}
        for entity in entities:
            category = entity.soap_category
            if category not in soap_groups:
                soap_groups[category] = []
            soap_groups[category].append(entity)
        
        # Infer relationships within and across SOAP categories
        relationship_rules = {
            (SOAPCategory.SUBJECTIVE, SOAPCategory.ASSESSMENT): RelationType.INDICATES,
            (SOAPCategory.OBJECTIVE, SOAPCategory.ASSESSMENT): RelationType.INDICATES,
            (SOAPCategory.ASSESSMENT, SOAPCategory.PLAN): RelationType.TREATS,
            (SOAPCategory.PLAN, SOAPCategory.ASSESSMENT): RelationType.TREATS,
        }
        
        for (cat1, cat2), relation_type in relationship_rules.items():
            entities1 = soap_groups.get(cat1, [])
            entities2 = soap_groups.get(cat2, [])
            
            for e1 in entities1:
                for e2 in entities2:
                    # Simple co-occurrence relationship
                    relationships.append({
                        'source': e1.text,
                        'target': e2.text,
                        'relation': relation_type.value.upper(),
                        'confidence': 0.5,
                        'soap_inference': f"{cat1.value}_to_{cat2.value}"
                    })
        
        return relationships
    
    def convert_to_medical_relations(self, relationships: List[Dict], 
                                   entities: List[MedicalEntity],
                                   soap_context: SOAPCategory = SOAPCategory.OBJECTIVE) -> List[MedicalRelation]:
        """Convert raw relationships to MedicalRelation objects"""
        medical_relations = []
        
        # Create entity lookup
        entity_by_text = {e.text.lower(): e.id for e in entities}
        
        for rel in relationships:
            try:
                source_text = rel.get('source', '').lower()
                target_text = rel.get('target', '').lower()
                
                # Find matching entities
                source_id = entity_by_text.get(source_text)
                target_id = entity_by_text.get(target_text)
                
                if not source_id or not target_id:
                    continue
                
                # Map relation type
                relation_type_str = rel.get('relation', 'TREATS').upper()
                try:
                    relation_type = RelationType(relation_type_str.lower())
                except ValueError:
                    relation_type = RelationType.TREATS
                
                medical_relation = MedicalRelation(
                    id=str(uuid.uuid4()),
                    source_entity=source_id,
                    target_entity=target_id,
                    relation_type=relation_type,
                    confidence=rel.get('confidence', 0.5),
                    soap_context=soap_context,
                    metadata={
                        'extraction_method': 'llm' if rel.get('confidence', 0) > 0.8 else 'rules',
                        **{k: v for k, v in rel.items() if k not in ['source', 'target', 'relation', 'confidence']}
                    }
                )
                
                medical_relations.append(medical_relation)
            
            except Exception as e:
                logger.error(f"Error converting relationship {rel}: {e}")
                continue
        
        return medical_relations
    
    def extract_domain_specific_relationships(self, entities: List[MedicalEntity]) -> List[Dict]:
        """Extract domain-specific medical relationships"""
        relationships = []
        
        # Group entities by type
        entity_groups = {}
        for entity in entities:
            entity_type = entity.entity_type
            if entity_type not in entity_groups:
                entity_groups[entity_type] = []
            entity_groups[entity_type].append(entity)
        
        # Define medical domain relationship rules
        domain_rules = [
            # Medications treat diseases
            ('medication', 'disease', RelationType.TREATS),
            # Procedures treat diseases  
            ('procedure', 'disease', RelationType.TREATS),
            # Diseases cause symptoms
            ('disease', 'symptom', RelationType.CAUSES),
            # Lab values indicate diseases
            ('lab_value', 'disease', RelationType.INDICATES),
            # Vital signs indicate conditions
            ('vital_sign', 'disease', RelationType.INDICATES),
            # Procedures examine anatomy
            ('procedure', 'anatomy', RelationType.LOCATED_IN),
        ]
        
        for source_type, target_type, relation_type in domain_rules:
            source_entities = entity_groups.get(source_type, [])
            target_entities = entity_groups.get(target_type, [])
            
            for source_entity in source_entities:
                for target_entity in target_entities:
                    relationships.append({
                        'source': source_entity.text,
                        'target': target_entity.text,
                        'relation': relation_type.value.upper(),
                        'confidence': 0.6,
                        'domain_rule': f"{source_type}_{relation_type.value}_{target_type}"
                    })
        
        return relationships
    
    def extract_relationships(self, text: str, entities: List[MedicalEntity], 
                            use_llm: bool = True) -> List[MedicalRelation]:
        """Main relationship extraction pipeline"""
        logger.info(f"Extracting relationships from {len(entities)} entities")
        
        relationship_sources = []
        
        # Extract using LLM if available
        if use_llm and entities:
            llm_relationships = self.extract_relationships_llm(text, entities)
            if llm_relationships:
                relationship_sources.append(llm_relationships)
        
        # Extract using rule-based methods
        rule_relationships = self.extract_relationships_rules(text, entities)
        relationship_sources.append(rule_relationships)
        
        # Extract co-occurrence relationships
        cooccurrence_relationships = self.extract_cooccurrence_relationships(entities)
        relationship_sources.append(cooccurrence_relationships)
        
        # Extract domain-specific relationships
        domain_relationships = self.extract_domain_specific_relationships(entities)
        relationship_sources.append(domain_relationships)
        
        # Merge all relationships
        all_relationships = []
        for rel_list in relationship_sources:
            all_relationships.extend(rel_list)
        
        # Remove duplicates based on source-target-relation triplets
        unique_relationships = []
        seen_triplets = set()
        
        for rel in all_relationships:
            # Safely handle None values
            source = rel.get('source', '') or ''
            target = rel.get('target', '') or ''
            relation = rel.get('relation', '') or ''
            
            triplet = (
                source.lower() if isinstance(source, str) else str(source).lower(),
                target.lower() if isinstance(target, str) else str(target).lower(),
                relation.lower() if isinstance(relation, str) else str(relation).lower()
            )
            
            if triplet not in seen_triplets:
                seen_triplets.add(triplet)
                unique_relationships.append(rel)
        
        # Convert to MedicalRelation objects
        medical_relations = self.convert_to_medical_relations(unique_relationships, entities)
        
        logger.info(f"Extracted {len(medical_relations)} unique relationships")
        return medical_relations