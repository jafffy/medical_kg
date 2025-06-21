import pytest
from unittest.mock import Mock, patch, MagicMock
from soap_kg.core.medical_ner import MedicalNER
from soap_kg.models.soap_schema import MedicalEntity, EntityType, SOAPCategory
from soap_kg.utils.openrouter_client import OpenRouterClient
from soap_kg.utils.text_preprocessor import MedicalTextPreprocessor


class TestMedicalNER:
    
    @pytest.fixture
    def mock_openrouter_client(self):
        """Create a mock OpenRouter client"""
        client = Mock(spec=OpenRouterClient)
        return client
    
    @pytest.fixture
    def ner_with_mock_client(self, mock_openrouter_client):
        """Create MedicalNER with mock client"""
        return MedicalNER(openrouter_client=mock_openrouter_client)
    
    @pytest.fixture
    def ner_default(self):
        """Create MedicalNER with default client"""
        with patch('soap_kg.core.medical_ner.OpenRouterClient'):
            return MedicalNER()
    
    def test_init_with_custom_client(self, mock_openrouter_client):
        """Test initialization with custom OpenRouter client"""
        ner = MedicalNER(openrouter_client=mock_openrouter_client)
        assert ner.client == mock_openrouter_client
        assert isinstance(ner.preprocessor, MedicalTextPreprocessor)
        assert hasattr(ner, 'entity_patterns')
    
    @patch('soap_kg.core.medical_ner.OpenRouterClient')
    def test_init_with_default_client(self, mock_openrouter_class):
        """Test initialization with default client"""
        mock_client = Mock()
        mock_openrouter_class.return_value = mock_client
        
        ner = MedicalNER()
        assert ner.client == mock_client
        mock_openrouter_class.assert_called_once()
    
    def test_entity_patterns_structure(self, ner_default):
        """Test that entity patterns are properly structured"""
        patterns = ner_default.entity_patterns
        
        # Check all expected entity types are present
        expected_types = [
            EntityType.DISEASE, EntityType.MEDICATION, EntityType.PROCEDURE,
            EntityType.ANATOMY, EntityType.VITAL_SIGN, EntityType.LAB_VALUE
        ]
        
        for entity_type in expected_types:
            assert entity_type in patterns
            assert isinstance(patterns[entity_type], list)
            assert len(patterns[entity_type]) > 0
            
            # Check each pattern is a string
            for pattern in patterns[entity_type]:
                assert isinstance(pattern, str)
                assert len(pattern) > 0
    
    def test_extract_entities_llm_success(self, ner_with_mock_client):
        """Test successful LLM entity extraction"""
        mock_response = [
            {
                "text": "hypertension",
                "type": "DISEASE",
                "confidence": 0.95,
                "start": 0,
                "end": 12
            },
            {
                "text": "aspirin",
                "type": "MEDICATION", 
                "confidence": 0.88,
                "start": 20,
                "end": 27
            }
        ]
        
        ner_with_mock_client.client.extract_entities.return_value = mock_response
        
        text = "hypertension treated with aspirin"
        result = ner_with_mock_client.extract_entities_llm(text)
        
        assert result == mock_response
        ner_with_mock_client.client.extract_entities.assert_called_once_with(text)
    
    def test_extract_entities_llm_failure(self, ner_with_mock_client):
        """Test LLM entity extraction failure handling"""
        ner_with_mock_client.client.extract_entities.side_effect = Exception("API Error")
        
        text = "hypertension treated with aspirin"
        result = ner_with_mock_client.extract_entities_llm(text)
        
        assert result == []
    
    def test_extract_entities_rule_based_diseases(self, ner_default):
        """Test rule-based extraction for diseases"""
        text = "Patient has hypertension and diabetes"
        
        with patch.object(ner_default, '_apply_entity_patterns') as mock_apply:
            mock_apply.return_value = [
                ("hypertension", EntityType.DISEASE, 12, 24),
                ("diabetes", EntityType.DISEASE, 29, 37)
            ]
            
            result = ner_default.extract_entities_rule_based(text)
            
            assert len(result) == 2
            assert all(isinstance(entity, MedicalEntity) for entity in result)
            assert result[0].text == "hypertension"
            assert result[0].entity_type == EntityType.DISEASE
    
    def test_extract_entities_rule_based_medications(self, ner_default):
        """Test rule-based extraction for medications"""
        text = "Prescribed aspirin and metformin"
        
        with patch.object(ner_default, '_apply_entity_patterns') as mock_apply:
            mock_apply.return_value = [
                ("aspirin", EntityType.MEDICATION, 11, 18),
                ("metformin", EntityType.MEDICATION, 23, 32)
            ]
            
            result = ner_default.extract_entities_rule_based(text)
            
            assert len(result) == 2
            assert result[0].entity_type == EntityType.MEDICATION
            assert result[1].entity_type == EntityType.MEDICATION
    
    def test_extract_entities_rule_based_empty_text(self, ner_default):
        """Test rule-based extraction with empty text"""
        result = ner_default.extract_entities_rule_based("")
        assert result == []
        
        result = ner_default.extract_entities_rule_based(None)
        assert result == []
    
    def test_extract_entities_hybrid_llm_priority(self, ner_with_mock_client):
        """Test hybrid extraction prioritizes LLM results"""
        llm_response = [
            {
                "text": "hypertension",
                "type": "DISEASE",
                "confidence": 0.95,
                "start": 0,
                "end": 12
            }
        ]
        
        ner_with_mock_client.client.extract_entities.return_value = llm_response
        
        with patch.object(ner_with_mock_client, 'extract_entities_rule_based') as mock_rule:
            mock_rule.return_value = [
                MedicalEntity(
                    text="diabetes",
                    entity_type=EntityType.DISEASE,
                    start_pos=15,
                    end_pos=23,
                    confidence=0.7
                )
            ]
            
            text = "hypertension and diabetes"
            result = ner_with_mock_client.extract_entities_hybrid(text)
            
            # Should have both LLM and rule-based results
            assert len(result) >= 2
            
            # First result should be from LLM (converted to MedicalEntity)
            llm_entity = next((e for e in result if e.text == "hypertension"), None)
            assert llm_entity is not None
            assert llm_entity.confidence == 0.95
    
    def test_extract_entities_hybrid_fallback_to_rules(self, ner_with_mock_client):
        """Test hybrid extraction falls back to rules when LLM fails"""
        ner_with_mock_client.client.extract_entities.side_effect = Exception("API Error")
        
        with patch.object(ner_with_mock_client, 'extract_entities_rule_based') as mock_rule:
            mock_entities = [
                MedicalEntity(
                    text="hypertension",
                    entity_type=EntityType.DISEASE,
                    start_pos=0,
                    end_pos=12,
                    confidence=0.8
                )
            ]
            mock_rule.return_value = mock_entities
            
            text = "hypertension"
            result = ner_with_mock_client.extract_entities_hybrid(text)
            
            assert result == mock_entities
            mock_rule.assert_called_once_with(text)
    
    def test_categorize_entities_by_soap(self, ner_default):
        """Test SOAP categorization of entities"""
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
                start_pos=15,
                end_pos=22,
                confidence=0.85
            )
        ]
        
        with patch.object(ner_default, '_categorize_entity_soap') as mock_categorize:
            mock_categorize.side_effect = [SOAPCategory.ASSESSMENT, SOAPCategory.PLAN]
            
            result = ner_default.categorize_entities_by_soap(entities)
            
            assert isinstance(result, dict)
            assert SOAPCategory.ASSESSMENT in result
            assert SOAPCategory.PLAN in result
            assert len(result[SOAPCategory.ASSESSMENT]) == 1
            assert len(result[SOAPCategory.PLAN]) == 1
    
    def test_categorize_entity_soap_disease(self, ner_default):
        """Test SOAP categorization for disease entities"""
        entity = MedicalEntity(
            text="hypertension",
            entity_type=EntityType.DISEASE,
            start_pos=0,
            end_pos=12,
            confidence=0.9
        )
        
        # Diseases typically go to Assessment
        result = ner_default._categorize_entity_soap(entity)
        assert result == SOAPCategory.ASSESSMENT
    
    def test_categorize_entity_soap_medication(self, ner_default):
        """Test SOAP categorization for medication entities"""
        entity = MedicalEntity(
            text="aspirin",
            entity_type=EntityType.MEDICATION,
            start_pos=0,
            end_pos=7,
            confidence=0.9
        )
        
        # Medications typically go to Plan
        result = ner_default._categorize_entity_soap(entity)
        assert result == SOAPCategory.PLAN
    
    def test_categorize_entity_soap_vital_sign(self, ner_default):
        """Test SOAP categorization for vital sign entities"""
        entity = MedicalEntity(
            text="blood pressure",
            entity_type=EntityType.VITAL_SIGN,
            start_pos=0,
            end_pos=14,
            confidence=0.9
        )
        
        # Vital signs typically go to Objective
        result = ner_default._categorize_entity_soap(entity)
        assert result == SOAPCategory.OBJECTIVE
    
    def test_process_clinical_text(self, ner_with_mock_client):
        """Test complete clinical text processing"""
        text = "Patient complains of chest pain. BP 140/90. Diagnosed with hypertension. Started on lisinopril."
        
        mock_entities = [
            {
                "text": "chest pain",
                "type": "DISEASE",
                "confidence": 0.9,
                "start": 21,
                "end": 31
            },
            {
                "text": "hypertension",
                "type": "DISEASE", 
                "confidence": 0.95,
                "start": 60,
                "end": 72
            },
            {
                "text": "lisinopril",
                "type": "MEDICATION",
                "confidence": 0.88,
                "start": 85,
                "end": 95
            }
        ]
        
        ner_with_mock_client.client.extract_entities.return_value = mock_entities
        
        result = ner_with_mock_client.process_clinical_text(text)
        
        assert 'entities' in result
        assert 'soap_categories' in result
        assert len(result['entities']) == 3
        assert isinstance(result['soap_categories'], dict)