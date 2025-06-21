"""
Simple tests to verify core functionality works.
"""

import pytest
from soap_kg.models.soap_schema import (
    MedicalEntity, EntityType, SOAPCategory, 
    MedicalRelation, RelationType, SOAPKnowledgeGraph
)


class TestSchemaModels:
    """Test the schema models work correctly"""
    
    def test_medical_entity_creation(self):
        """Test creating a medical entity"""
        entity = MedicalEntity(
            id="test_1",
            text="hypertension",
            entity_type=EntityType.DISEASE,
            soap_category=SOAPCategory.ASSESSMENT,
            confidence=0.95
        )
        
        assert entity.id == "test_1"
        assert entity.text == "hypertension"
        assert entity.entity_type == EntityType.DISEASE
        assert entity.soap_category == SOAPCategory.ASSESSMENT
        assert entity.confidence == 0.95
        assert entity.metadata == {}
    
    def test_medical_relation_creation(self):
        """Test creating a medical relation"""
        relation = MedicalRelation(
            id="rel_1",
            source_entity="ent_1",
            target_entity="ent_2",
            relation_type=RelationType.TREATS,
            confidence=0.88,
            soap_context=SOAPCategory.PLAN
        )
        
        assert relation.id == "rel_1"
        assert relation.source_entity == "ent_1"
        assert relation.target_entity == "ent_2"
        assert relation.relation_type == RelationType.TREATS
        assert relation.confidence == 0.88
        assert relation.soap_context == SOAPCategory.PLAN
        assert relation.metadata == {}
    
    def test_knowledge_graph_basic_operations(self):
        """Test basic knowledge graph operations"""
        kg = SOAPKnowledgeGraph()
        
        # Create entities
        entity1 = MedicalEntity(
            id="ent_1",
            text="hypertension",
            entity_type=EntityType.DISEASE,
            soap_category=SOAPCategory.ASSESSMENT,
            confidence=0.95
        )
        
        entity2 = MedicalEntity(
            id="ent_2",
            text="aspirin",
            entity_type=EntityType.MEDICATION,
            soap_category=SOAPCategory.PLAN,
            confidence=0.90
        )
        
        # Add entities
        kg.add_entity(entity1)
        kg.add_entity(entity2)
        
        assert len(kg.entities) == 2
        assert "ent_1" in kg.entities
        assert "ent_2" in kg.entities
        
        # Create and add relation
        relation = MedicalRelation(
            id="rel_1",
            source_entity="ent_2",
            target_entity="ent_1",
            relation_type=RelationType.TREATS,
            confidence=0.88,
            soap_context=SOAPCategory.PLAN
        )
        
        kg.add_relation(relation)
        
        assert len(kg.relations) == 1
        assert "rel_1" in kg.relations
    
    def test_knowledge_graph_statistics(self):
        """Test knowledge graph statistics generation"""
        kg = SOAPKnowledgeGraph()
        
        # Add some entities
        for i, (text, etype, category) in enumerate([
            ("hypertension", EntityType.DISEASE, SOAPCategory.ASSESSMENT),
            ("aspirin", EntityType.MEDICATION, SOAPCategory.PLAN),
            ("chest pain", EntityType.SYMPTOM, SOAPCategory.SUBJECTIVE)
        ]):
            entity = MedicalEntity(
                id=f"ent_{i}",
                text=text,
                entity_type=etype,
                soap_category=category,
                confidence=0.9
            )
            kg.add_entity(entity)
        
        stats = kg.get_statistics()
        
        assert stats["total_entities"] == 3
        assert stats["total_relations"] == 0
        assert stats["total_patients"] == 0
        assert len(stats["entity_types"]) == 3
        assert EntityType.DISEASE in stats["entity_types"]
        assert EntityType.MEDICATION in stats["entity_types"]
        assert EntityType.SYMPTOM in stats["entity_types"]


class TestDataLoaderBasic:
    """Basic tests for data loader without external dependencies"""
    
    def test_data_loader_import(self):
        """Test that data loader can be imported"""
        from soap_kg.core.data_loader import MimicDataLoader
        
        # Should be able to create instance
        loader = MimicDataLoader(data_path="/tmp/test")
        assert loader.data_path == "/tmp/test"
        assert loader.tables == {}
    
    def test_data_loader_csv_not_found(self):
        """Test data loader handles missing files gracefully"""
        from soap_kg.core.data_loader import MimicDataLoader
        
        loader = MimicDataLoader(data_path="/nonexistent/path")
        result = loader.load_csv("nonexistent_table")
        
        # Should return empty DataFrame without crashing
        assert result.empty
        assert "nonexistent_table" not in loader.tables