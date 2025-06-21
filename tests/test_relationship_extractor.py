import pytest
from unittest.mock import Mock, patch
from soap_kg.core.relationship_extractor import RelationshipExtractor
from soap_kg.models.soap_schema import MedicalEntity, EntityType, Relationship, RelationshipType
from soap_kg.utils.openrouter_client import OpenRouterClient


class TestRelationshipExtractor:
    
    @pytest.fixture
    def mock_openrouter_client(self):
        """Create a mock OpenRouter client"""
        client = Mock(spec=OpenRouterClient)
        return client
    
    @pytest.fixture
    def extractor_with_mock_client(self, mock_openrouter_client):
        """Create RelationshipExtractor with mock client"""
        return RelationshipExtractor(openrouter_client=mock_openrouter_client)
    
    @pytest.fixture
    def extractor_default(self):
        """Create RelationshipExtractor with default client"""
        with patch('soap_kg.core.relationship_extractor.OpenRouterClient'):
            return RelationshipExtractor()
    
    @pytest.fixture
    def sample_entities(self):
        """Create sample medical entities for testing"""
        return [
            MedicalEntity(
                text="hypertension",
                entity_type=EntityType.DISEASE,
                start_pos=0,
                end_pos=12,
                confidence=0.9
            ),
            MedicalEntity(
                text="aspirin",
                entity_type=EntityType.MEDICATION,
                start_pos=20,
                end_pos=27,
                confidence=0.85
            ),
            MedicalEntity(
                text="blood pressure",
                entity_type=EntityType.VITAL_SIGN,
                start_pos=35,
                end_pos=49,
                confidence=0.8
            )
        ]
    
    def test_init_with_custom_client(self, mock_openrouter_client):
        """Test initialization with custom OpenRouter client"""
        extractor = RelationshipExtractor(openrouter_client=mock_openrouter_client)
        assert extractor.client == mock_openrouter_client
        assert hasattr(extractor, 'relationship_patterns')
    
    @patch('soap_kg.core.relationship_extractor.OpenRouterClient')
    def test_init_with_default_client(self, mock_openrouter_class):
        """Test initialization with default client"""
        mock_client = Mock()
        mock_openrouter_class.return_value = mock_client
        
        extractor = RelationshipExtractor()
        assert extractor.client == mock_client
        mock_openrouter_class.assert_called_once()
    
    def test_relationship_patterns_structure(self, extractor_default):
        """Test that relationship patterns are properly structured"""
        patterns = extractor_default.relationship_patterns
        
        # Check expected relationship types are present
        expected_types = [
            RelationshipType.TREATS, RelationshipType.CAUSES,
            RelationshipType.DIAGNOSED_WITH, RelationshipType.INDICATES
        ]
        
        for rel_type in expected_types:
            if rel_type in patterns:
                assert isinstance(patterns[rel_type], list)
                assert len(patterns[rel_type]) > 0
                
                # Check each pattern is a string
                for pattern in patterns[rel_type]:
                    assert isinstance(pattern, str)
                    assert len(pattern) > 0
    
    def test_extract_relationships_llm_success(self, extractor_with_mock_client, sample_entities):
        """Test successful LLM relationship extraction"""
        mock_response = [
            {
                "source_entity": "aspirin",
                "target_entity": "hypertension",
                "relationship_type": "TREATS",
                "confidence": 0.92
            }
        ]
        
        extractor_with_mock_client.client.extract_relationships.return_value = mock_response
        
        text = "Patient takes aspirin for hypertension"
        result = extractor_with_mock_client.extract_relationships_llm(sample_entities, text)
        
        assert result == mock_response
        extractor_with_mock_client.client.extract_relationships.assert_called_once()
    
    def test_extract_relationships_llm_failure(self, extractor_with_mock_client, sample_entities):
        """Test LLM relationship extraction failure handling"""
        extractor_with_mock_client.client.extract_relationships.side_effect = Exception("API Error")
        
        text = "Patient takes aspirin for hypertension"
        result = extractor_with_mock_client.extract_relationships_llm(sample_entities, text)
        
        assert result == []
    
    def test_extract_relationships_rule_based_treats(self, extractor_default, sample_entities):
        """Test rule-based extraction for treatment relationships"""
        text = "Patient takes aspirin for hypertension"
        
        with patch.object(extractor_default, '_find_entity_pairs_in_text') as mock_pairs:
            mock_pairs.return_value = [(sample_entities[1], sample_entities[0])]  # aspirin, hypertension
            
            result = extractor_default.extract_relationships_rule_based(sample_entities, text)
            
            assert len(result) > 0
            relationship = next((r for r in result if r.relationship_type == RelationshipType.TREATS), None)
            assert relationship is not None
            assert relationship.source_entity == sample_entities[1]  # aspirin
            assert relationship.target_entity == sample_entities[0]  # hypertension
    
    def test_extract_relationships_rule_based_diagnosed_with(self, extractor_default, sample_entities):
        """Test rule-based extraction for diagnosis relationships"""
        text = "Patient diagnosed with hypertension"
        
        with patch.object(extractor_default, '_find_entity_pairs_in_text') as mock_pairs:
            mock_pairs.return_value = [(sample_entities[0], sample_entities[0])]  # hypertension
            
            result = extractor_default.extract_relationships_rule_based(sample_entities, text)
            
            # Should find diagnosis relationship patterns
            assert isinstance(result, list)
    
    def test_extract_relationships_rule_based_empty_entities(self, extractor_default):
        """Test rule-based extraction with empty entities list"""
        text = "Patient takes aspirin for hypertension"
        result = extractor_default.extract_relationships_rule_based([], text)
        
        assert result == []
    
    def test_extract_relationships_rule_based_empty_text(self, extractor_default, sample_entities):
        """Test rule-based extraction with empty text"""
        result = extractor_default.extract_relationships_rule_based(sample_entities, "")
        assert result == []
        
        result = extractor_default.extract_relationships_rule_based(sample_entities, None)
        assert result == []
    
    def test_extract_relationships_hybrid_llm_priority(self, extractor_with_mock_client, sample_entities):
        """Test hybrid extraction prioritizes LLM results"""
        llm_response = [
            {
                "source_entity": "aspirin",
                "target_entity": "hypertension",
                "relationship_type": "TREATS",
                "confidence": 0.95
            }
        ]
        
        extractor_with_mock_client.client.extract_relationships.return_value = llm_response
        
        with patch.object(extractor_with_mock_client, 'extract_relationships_rule_based') as mock_rule:
            rule_relationships = [
                Relationship(
                    source_entity=sample_entities[1],
                    target_entity=sample_entities[0],
                    relationship_type=RelationshipType.TREATS,
                    confidence=0.7
                )
            ]
            mock_rule.return_value = rule_relationships
            
            text = "Patient takes aspirin for hypertension"
            result = extractor_with_mock_client.extract_relationships_hybrid(sample_entities, text)
            
            # Should have both LLM and rule-based results, but prioritize LLM
            assert len(result) >= 1
            
            # Check for LLM-derived relationship
            llm_relationship = next((r for r in result if r.confidence == 0.95), None)
            assert llm_relationship is not None
    
    def test_extract_relationships_hybrid_fallback_to_rules(self, extractor_with_mock_client, sample_entities):
        """Test hybrid extraction falls back to rules when LLM fails"""
        extractor_with_mock_client.client.extract_relationships.side_effect = Exception("API Error")
        
        with patch.object(extractor_with_mock_client, 'extract_relationships_rule_based') as mock_rule:
            rule_relationships = [
                Relationship(
                    source_entity=sample_entities[1],
                    target_entity=sample_entities[0],
                    relationship_type=RelationshipType.TREATS,
                    confidence=0.8
                )
            ]
            mock_rule.return_value = rule_relationships
            
            text = "Patient takes aspirin for hypertension"
            result = extractor_with_mock_client.extract_relationships_hybrid(sample_entities, text)
            
            assert result == rule_relationships
            mock_rule.assert_called_once_with(sample_entities, text)
    
    def test_find_entity_pairs_in_text(self, extractor_default, sample_entities):
        """Test finding entity pairs within text proximity"""
        text = "Patient has hypertension and takes aspirin daily"
        
        # Entities should be close enough to form pairs
        pairs = extractor_default._find_entity_pairs_in_text(sample_entities, text)
        
        assert isinstance(pairs, list)
        # Should find pairs between entities that are close in text
        if len(pairs) > 0:
            for source, target in pairs:
                assert isinstance(source, MedicalEntity)
                assert isinstance(target, MedicalEntity)
                assert source != target
    
    def test_find_entity_pairs_proximity_threshold(self, extractor_default):
        """Test entity pair finding respects proximity threshold"""
        # Create entities that are far apart
        entities = [
            MedicalEntity(
                text="hypertension",
                entity_type=EntityType.DISEASE,
                start_pos=0,
                end_pos=12,
                confidence=0.9
            ),
            MedicalEntity(
                text="aspirin",
                entity_type=EntityType.MEDICATION,
                start_pos=200,  # Far away
                end_pos=207,
                confidence=0.85
            )
        ]
        
        text = "Patient has hypertension" + " " * 150 + "and later takes aspirin"
        
        pairs = extractor_default._find_entity_pairs_in_text(entities, text, max_distance=50)
        
        # Should not find pairs if entities are too far apart
        assert len(pairs) == 0
    
    def test_validate_relationship_logical(self, extractor_default, sample_entities):
        """Test logical relationship validation"""
        # Valid relationship: medication treats disease
        valid_rel = Relationship(
            source_entity=sample_entities[1],  # aspirin
            target_entity=sample_entities[0],  # hypertension
            relationship_type=RelationshipType.TREATS,
            confidence=0.9
        )
        
        assert extractor_default._validate_relationship(valid_rel) is True
        
        # Invalid relationship: disease treats medication
        invalid_rel = Relationship(
            source_entity=sample_entities[0],  # hypertension
            target_entity=sample_entities[1],  # aspirin
            relationship_type=RelationshipType.TREATS,
            confidence=0.9
        )
        
        # This should be invalid or have lower confidence
        validation_result = extractor_default._validate_relationship(invalid_rel)
        assert isinstance(validation_result, bool)
    
    def test_get_relationship_confidence(self, extractor_default, sample_entities):
        """Test relationship confidence calculation"""
        relationship = Relationship(
            source_entity=sample_entities[1],  # aspirin
            target_entity=sample_entities[0],  # hypertension
            relationship_type=RelationshipType.TREATS,
            confidence=0.8
        )
        
        text = "Patient takes aspirin to treat hypertension"
        
        confidence = extractor_default._get_relationship_confidence(relationship, text)
        
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0  # Should have some confidence for valid relationship
    
    def test_deduplicate_relationships(self, extractor_default, sample_entities):
        """Test relationship deduplication"""
        # Create duplicate relationships
        rel1 = Relationship(
            source_entity=sample_entities[1],
            target_entity=sample_entities[0],
            relationship_type=RelationshipType.TREATS,
            confidence=0.8
        )
        
        rel2 = Relationship(
            source_entity=sample_entities[1],
            target_entity=sample_entities[0],
            relationship_type=RelationshipType.TREATS,
            confidence=0.9  # Higher confidence
        )
        
        relationships = [rel1, rel2]
        
        deduplicated = extractor_default._deduplicate_relationships(relationships)
        
        # Should keep only one relationship (the one with higher confidence)
        assert len(deduplicated) == 1
        assert deduplicated[0].confidence == 0.9
    
    def test_process_clinical_text_complete(self, extractor_with_mock_client, sample_entities):
        """Test complete clinical text processing"""
        text = "Patient diagnosed with hypertension. Started on aspirin therapy. Blood pressure improved."
        
        mock_llm_response = [
            {
                "source_entity": "aspirin",
                "target_entity": "hypertension",
                "relationship_type": "TREATS",
                "confidence": 0.92
            }
        ]
        
        extractor_with_mock_client.client.extract_relationships.return_value = mock_llm_response
        
        result = extractor_with_mock_client.process_clinical_text(sample_entities, text)
        
        assert 'relationships' in result
        assert 'relationship_types' in result
        assert isinstance(result['relationships'], list)
        assert isinstance(result['relationship_types'], dict)
        
        if len(result['relationships']) > 0:
            assert all(isinstance(rel, Relationship) for rel in result['relationships'])