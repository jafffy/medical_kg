from typing import List, Dict, Optional, Tuple, Set
import logging
import time
import itertools
from collections import defaultdict
from soap_kg.utils.openrouter_client import OpenRouterClient
from soap_kg.models.soap_schema import MedicalEntity, MedicalRelation, RelationType, SOAPCategory
from soap_kg.config import Config
import uuid
import re

logger = logging.getLogger(__name__)

class RelationshipExtractor:
    def __init__(self, openrouter_client: OpenRouterClient = None):
        self.client = openrouter_client or OpenRouterClient()
        
        # Performance monitoring
        self.performance_stats = {
            'total_extractions': 0,
            'total_entities_processed': 0,
            'total_relationships_found': 0,
            'avg_extraction_time': 0.0,
            'cache_hits': 0
        }
        
        # Entity indexing for performance
        self._entity_index = {}  # text -> entity mapping
        self._entity_pairs_cache = {}  # Cache for entity pairs
        self._pattern_cache = {}  # Cache for compiled patterns
        
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
    
    def _build_entity_index(self, entities: List[MedicalEntity]) -> Dict:
        """Build optimized entity index for fast lookups"""
        if not Config.ENABLE_ENTITY_INDEXING:
            return {}
            
        index = {
            'by_text': {e.text.lower(): e for e in entities},
            'by_type': defaultdict(list),
            'by_soap_category': defaultdict(list),
            'positions': {}  # For text position-based optimization
        }
        
        for entity in entities:
            index['by_type'][entity.entity_type].append(entity)
            index['by_soap_category'][entity.soap_category].append(entity)
            
        return index
    
    def _get_entity_pairs_optimized(self, entities: List[MedicalEntity], max_pairs: int = None) -> List[Tuple[MedicalEntity, MedicalEntity]]:
        """Get entity pairs with optimization and limits"""
        max_pairs = max_pairs or Config.MAX_ENTITY_PAIRS_PER_BATCH
        
        # Limit entities if too many
        if len(entities) > Config.MAX_ENTITIES_FOR_RELATIONSHIP_EXTRACTION:
            logger.warning(f"Too many entities ({len(entities)}), limiting to {Config.MAX_ENTITIES_FOR_RELATIONSHIP_EXTRACTION}")
            # Sort by confidence and take top entities
            entities = sorted(entities, key=lambda e: e.confidence, reverse=True)[:Config.MAX_ENTITIES_FOR_RELATIONSHIP_EXTRACTION]
        
        # Generate pairs efficiently
        pairs = []
        pair_count = 0
        
        for i, entity1 in enumerate(entities):
            for j in range(i + 1, len(entities)):
                if pair_count >= max_pairs:
                    logger.info(f"Reached maximum entity pairs limit: {max_pairs}")
                    break
                    
                entity2 = entities[j]
                pairs.append((entity1, entity2))
                pairs.append((entity2, entity1))  # Both directions
                pair_count += 2
                
            if pair_count >= max_pairs:
                break
                
        return pairs
    
    def extract_relationships_rules(self, text: str, entities: List[MedicalEntity]) -> List[Dict]:
        """Optimized rule-based relationship extraction"""
        start_time = time.time()
        relationships = []
        
        if not entities:
            return relationships
            
        # Build entity index for fast lookups
        entity_index = self._build_entity_index(entities)
        
        # Get optimized entity pairs
        entity_pairs = self._get_entity_pairs_optimized(entities)
        
        logger.info(f"Processing {len(entity_pairs)} entity pairs for rule-based extraction")
        
        # Limit patterns per type for performance
        max_patterns = Config.MAX_RELATIONSHIP_PATTERNS_PER_TYPE
        
        for relation_type, patterns in self.relation_patterns.items():
            # Limit number of patterns to avoid performance issues
            limited_patterns = patterns[:max_patterns]
            
            for pattern_template in limited_patterns:
                for entity1, entity2 in entity_pairs:
                    # Skip if same entity
                    if entity1.id == entity2.id:
                        continue
                    
                    # Create pattern cache key
                    cache_key = (pattern_template, entity1.text, entity2.text)
                    
                    if cache_key in self._pattern_cache:
                        pattern = self._pattern_cache[cache_key]
                        self.performance_stats['cache_hits'] += 1
                    else:
                        # Create pattern with specific entities
                        pattern = pattern_template.format(
                            entity1=re.escape(entity1.text),
                            entity2=re.escape(entity2.text)
                        )
                        self._pattern_cache[cache_key] = pattern
                    
                    try:
                        matches = re.search(pattern, text, re.IGNORECASE)
                        if matches:
                            relationships.append({
                                'source': entity1.text,
                                'target': entity2.text,
                                'relation': relation_type.value.upper(),
                                'confidence': 0.7,
                                'pattern_match': matches.group(0),
                                'extraction_method': 'rules_optimized'
                            })
                    except re.error as e:
                        logger.warning(f"Invalid regex pattern {pattern}: {e}")
                        continue
        
        extraction_time = time.time() - start_time
        if Config.ENABLE_PERFORMANCE_MONITORING:
            logger.info(f"Rule-based extraction took {extraction_time:.3f}s for {len(entities)} entities")
            
        return relationships
    
    def extract_cooccurrence_relationships(self, entities: List[MedicalEntity], 
                                         window_size: int = None) -> List[Dict]:
        """Optimized entity co-occurrence relationship extraction"""
        window_size = window_size or Config.RELATIONSHIP_WINDOW_SIZE
        relationships = []
        
        if not entities:
            return relationships
            
        # Limit entities for performance
        if len(entities) > Config.MAX_ENTITIES_FOR_RELATIONSHIP_EXTRACTION:
            entities = sorted(entities, key=lambda e: e.confidence, reverse=True)[:Config.MAX_ENTITIES_FOR_RELATIONSHIP_EXTRACTION]
        
        # Group entities by SOAP category for better relationship inference
        soap_groups = defaultdict(list)
        for entity in entities:
            soap_groups[entity.soap_category].append(entity)
        
        # Optimized relationship rules with limits
        relationship_rules = {
            (SOAPCategory.SUBJECTIVE, SOAPCategory.ASSESSMENT): RelationType.INDICATES,
            (SOAPCategory.OBJECTIVE, SOAPCategory.ASSESSMENT): RelationType.INDICATES,
            (SOAPCategory.ASSESSMENT, SOAPCategory.PLAN): RelationType.TREATS,
            (SOAPCategory.PLAN, SOAPCategory.ASSESSMENT): RelationType.TREATS,
        }
        
        total_pairs = 0
        max_pairs = Config.MAX_ENTITY_PAIRS_PER_BATCH
        
        for (cat1, cat2), relation_type in relationship_rules.items():
            entities1 = soap_groups.get(cat1, [])
            entities2 = soap_groups.get(cat2, [])
            
            # Limit pairs to avoid performance issues
            for e1 in entities1:
                for e2 in entities2:
                    if total_pairs >= max_pairs:
                        logger.warning(f"Reached maximum co-occurrence pairs limit: {max_pairs}")
                        break
                        
                    # Simple co-occurrence relationship
                    relationships.append({
                        'source': e1.text,
                        'target': e2.text,
                        'relation': relation_type.value.upper(),
                        'confidence': 0.5,
                        'soap_inference': f"{cat1.value}_to_{cat2.value}",
                        'extraction_method': 'cooccurrence_optimized'
                    })
                    total_pairs += 1
                    
                if total_pairs >= max_pairs:
                    break
        
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
        """Optimized domain-specific medical relationship extraction"""
        relationships = []
        
        if not entities:
            return relationships
            
        # Limit entities for performance
        if len(entities) > Config.MAX_ENTITIES_FOR_RELATIONSHIP_EXTRACTION:
            entities = sorted(entities, key=lambda e: e.confidence, reverse=True)[:Config.MAX_ENTITIES_FOR_RELATIONSHIP_EXTRACTION]
        
        # Group entities by type using defaultdict for efficiency
        entity_groups = defaultdict(list)
        for entity in entities:
            entity_groups[entity.entity_type].append(entity)
        
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
        
        total_pairs = 0
        max_pairs = Config.MAX_ENTITY_PAIRS_PER_BATCH
        
        for source_type_str, target_type_str, relation_type in domain_rules:
            # Convert string to enum for lookup
            try:
                from soap_kg.models.soap_schema import EntityType
                source_type = EntityType(source_type_str)
                target_type = EntityType(target_type_str)
            except ValueError:
                continue
                
            source_entities = entity_groups.get(source_type, [])
            target_entities = entity_groups.get(target_type, [])
            
            for source_entity in source_entities:
                for target_entity in target_entities:
                    if total_pairs >= max_pairs:
                        logger.warning(f"Reached maximum domain-specific pairs limit: {max_pairs}")
                        break
                        
                    relationships.append({
                        'source': source_entity.text,
                        'target': target_entity.text,
                        'relation': relation_type.value.upper(),
                        'confidence': 0.6,
                        'domain_rule': f"{source_type_str}_{relation_type.value}_{target_type_str}",
                        'extraction_method': 'domain_specific_optimized'
                    })
                    total_pairs += 1
                    
                if total_pairs >= max_pairs:
                    break
                    
            if total_pairs >= max_pairs:
                break
        
        return relationships
    
    def _deduplicate_relationships_optimized(self, relationships: List[Dict]) -> List[Dict]:
        """Optimized relationship deduplication using sets"""
        if not relationships:
            return []
            
        # Use dict to maintain order while deduplicating
        unique_rels = {}
        
        for rel in relationships:
            # Safely handle None values
            source = rel.get('source', '') or ''
            target = rel.get('target', '') or ''
            relation = rel.get('relation', '') or ''
            
            # Create normalized triplet key
            triplet = (
                source.lower().strip() if isinstance(source, str) else str(source).lower().strip(),
                target.lower().strip() if isinstance(target, str) else str(target).lower().strip(),
                relation.lower().strip() if isinstance(relation, str) else str(relation).lower().strip()
            )
            
            # Skip empty or invalid triplets
            if not all(triplet) or triplet[0] == triplet[1]:
                continue
                
            # Keep the relationship with highest confidence for duplicates
            if triplet not in unique_rels or rel.get('confidence', 0) > unique_rels[triplet].get('confidence', 0):
                unique_rels[triplet] = rel
        
        return list(unique_rels.values())
    
    def extract_relationships(self, text: str, entities: List[MedicalEntity], 
                            use_llm: bool = True) -> List[MedicalRelation]:
        """Optimized main relationship extraction pipeline"""
        start_time = time.time()
        
        # Performance check
        if not entities:
            logger.info("No entities provided for relationship extraction")
            return []
            
        if len(entities) > Config.MAX_ENTITIES_FOR_RELATIONSHIP_EXTRACTION * 2:
            logger.warning(f"Very large entity set ({len(entities)}), this may impact performance")
        
        logger.info(f"Extracting relationships from {len(entities)} entities")
        
        # Update performance stats
        self.performance_stats['total_extractions'] += 1
        self.performance_stats['total_entities_processed'] += len(entities)
        
        relationship_sources = []
        
        # Extract using LLM if available (highest priority)
        if use_llm and entities:
            llm_start = time.time()
            llm_relationships = self.extract_relationships_llm(text, entities)
            if llm_relationships:
                relationship_sources.append(llm_relationships)
                logger.info(f"LLM extraction took {time.time() - llm_start:.3f}s, found {len(llm_relationships)} relationships")
        
        # Extract using rule-based methods (optimized)
        rule_start = time.time()
        rule_relationships = self.extract_relationships_rules(text, entities)
        relationship_sources.append(rule_relationships)
        logger.info(f"Rule-based extraction took {time.time() - rule_start:.3f}s, found {len(rule_relationships)} relationships")
        
        # Extract co-occurrence relationships (optimized)
        cooc_start = time.time()
        cooccurrence_relationships = self.extract_cooccurrence_relationships(entities)
        relationship_sources.append(cooccurrence_relationships)
        logger.info(f"Co-occurrence extraction took {time.time() - cooc_start:.3f}s, found {len(cooccurrence_relationships)} relationships")
        
        # Extract domain-specific relationships (optimized)
        domain_start = time.time()
        domain_relationships = self.extract_domain_specific_relationships(entities)
        relationship_sources.append(domain_relationships)
        logger.info(f"Domain-specific extraction took {time.time() - domain_start:.3f}s, found {len(domain_relationships)} relationships")
        
        # Merge all relationships efficiently
        all_relationships = []
        for rel_list in relationship_sources:
            all_relationships.extend(rel_list)
        
        logger.info(f"Total relationships before deduplication: {len(all_relationships)}")
        
        # Optimized deduplication
        dedup_start = time.time()
        unique_relationships = self._deduplicate_relationships_optimized(all_relationships)
        logger.info(f"Deduplication took {time.time() - dedup_start:.3f}s, {len(unique_relationships)} unique relationships")
        
        # Convert to MedicalRelation objects
        convert_start = time.time()
        medical_relations = self.convert_to_medical_relations(unique_relationships, entities)
        logger.info(f"Conversion took {time.time() - convert_start:.3f}s")
        
        # Update performance statistics
        total_time = time.time() - start_time
        self.performance_stats['total_relationships_found'] += len(medical_relations)
        self.performance_stats['avg_extraction_time'] = (
            (self.performance_stats['avg_extraction_time'] * (self.performance_stats['total_extractions'] - 1) + total_time) /
            self.performance_stats['total_extractions']
        )
        
        if Config.ENABLE_PERFORMANCE_MONITORING:
            logger.info(f"Total extraction time: {total_time:.3f}s for {len(entities)} entities -> {len(medical_relations)} relationships")
            logger.info(f"Performance stats: {self.performance_stats}")
        
        return medical_relations
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics for monitoring"""
        return self.performance_stats.copy()
    
    def reset_performance_stats(self):
        """Reset performance statistics"""
        self.performance_stats = {
            'total_extractions': 0,
            'total_entities_processed': 0,
            'total_relationships_found': 0,
            'avg_extraction_time': 0.0,
            'cache_hits': 0
        }