import pytest
from unittest.mock import Mock, patch
from soap_kg.core.soap_categorizer import SOAPCategorizer
from soap_kg.models.soap_schema import MedicalEntity, SOAPCategory, EntityType
from soap_kg.utils.openrouter_client import OpenRouterClient


class TestSOAPCategorizer:
    
    @pytest.fixture
    def mock_openrouter_client(self):
        """Create a mock OpenRouter client"""
        client = Mock(spec=OpenRouterClient)
        return client
    
    @pytest.fixture
    def categorizer_with_mock_client(self, mock_openrouter_client):
        """Create SOAPCategorizer with mock client"""
        return SOAPCategorizer(openrouter_client=mock_openrouter_client)
    
    @pytest.fixture
    def categorizer_default(self):
        """Create SOAPCategorizer with default client"""
        with patch('soap_kg.core.soap_categorizer.OpenRouterClient'):
            return SOAPCategorizer()
    
    def test_init_with_custom_client(self, mock_openrouter_client):
        """Test initialization with custom OpenRouter client"""
        categorizer = SOAPCategorizer(openrouter_client=mock_openrouter_client)
        assert categorizer.client == mock_openrouter_client
        assert hasattr(categorizer, 'soap_patterns')
        assert hasattr(categorizer, 'entity_soap_mapping')
    
    @patch('soap_kg.core.soap_categorizer.OpenRouterClient')
    def test_init_with_default_client(self, mock_openrouter_class):
        """Test initialization with default client"""
        mock_client = Mock()
        mock_openrouter_class.return_value = mock_client
        
        categorizer = SOAPCategorizer()
        assert categorizer.client == mock_client
        mock_openrouter_class.assert_called_once()
    
    def test_soap_patterns_structure(self, categorizer_default):
        """Test that SOAP patterns are properly structured"""
        patterns = categorizer_default.soap_patterns
        
        # Check all SOAP categories are present
        expected_categories = [
            SOAPCategory.SUBJECTIVE, SOAPCategory.OBJECTIVE,
            SOAPCategory.ASSESSMENT, SOAPCategory.PLAN
        ]
        
        for category in expected_categories:
            assert category in patterns
            assert isinstance(patterns[category], list)
            assert len(patterns[category]) > 0
            
            # Check each pattern is a string
            for pattern in patterns[category]:
                assert isinstance(pattern, str)
                assert len(pattern) > 0
    
    def test_entity_soap_mapping_structure(self, categorizer_default):
        """Test entity to SOAP mapping structure"""
        mapping = categorizer_default.entity_soap_mapping
        
        # Check expected entity types are mapped
        expected_mappings = {
            EntityType.SYMPTOM: SOAPCategory.SUBJECTIVE,
            EntityType.VITAL_SIGN: SOAPCategory.OBJECTIVE,
            EntityType.LAB_VALUE: SOAPCategory.OBJECTIVE,
            EntityType.DISEASE: SOAPCategory.ASSESSMENT,
            EntityType.MEDICATION: SOAPCategory.PLAN
        }
        
        for entity_type, expected_category in expected_mappings.items():
            assert entity_type in mapping
            assert mapping[entity_type] == expected_category
    
    def test_categorize_text_llm_success(self, categorizer_with_mock_client):
        """Test successful LLM text categorization"""
        mock_response = {
            "category": "SUBJECTIVE",
            "confidence": 0.92,
            "reasoning": "Patient is describing symptoms"
        }
        
        categorizer_with_mock_client.client.categorize_soap.return_value = mock_response
        
        text = "Patient complains of chest pain"
        result = categorizer_with_mock_client.categorize_text_llm(text)
        
        assert result == mock_response
        categorizer_with_mock_client.client.categorize_soap.assert_called_once_with(text)
    
    def test_categorize_text_llm_failure(self, categorizer_with_mock_client):
        """Test LLM text categorization failure handling"""
        categorizer_with_mock_client.client.categorize_soap.side_effect = Exception("API Error")
        
        text = "Patient complains of chest pain"
        result = categorizer_with_mock_client.categorize_text_llm(text)
        
        assert result is None
    
    def test_categorize_text_rule_based_subjective(self, categorizer_default):
        """Test rule-based categorization for subjective text"""
        texts = [
            "Patient complains of chest pain",
            "Patient reports feeling dizzy",
            "Chief complaint is nausea",
            "Patient denies shortness of breath"
        ]
        
        for text in texts:
            result = categorizer_default.categorize_text_rule_based(text)
            assert result == SOAPCategory.SUBJECTIVE
    
    def test_categorize_text_rule_based_objective(self, categorizer_default):
        """Test rule-based categorization for objective text"""
        texts = [
            "Vital signs: BP 120/80, HR 72",
            "Physical exam reveals normal heart sounds",
            "Lab results show elevated glucose",
            "CT scan shows no abnormalities"
        ]
        
        for text in texts:
            result = categorizer_default.categorize_text_rule_based(text)
            assert result == SOAPCategory.OBJECTIVE
    
    def test_categorize_text_rule_based_assessment(self, categorizer_default):
        """Test rule-based categorization for assessment text"""
        texts = [
            "Diagnosis: Hypertension",
            "Assessment shows stable condition",
            "Working diagnosis is pneumonia",
            "Rule out myocardial infarction"
        ]
        
        for text in texts:
            result = categorizer_default.categorize_text_rule_based(text)
            assert result == SOAPCategory.ASSESSMENT
    
    def test_categorize_text_rule_based_plan(self, categorizer_default):
        """Test rule-based categorization for plan text"""
        texts = [
            "Plan: Start aspirin 81mg daily",
            "Continue current medications",
            "Schedule follow-up in 2 weeks",
            "Patient education on diet"
        ]
        
        for text in texts:
            result = categorizer_default.categorize_text_rule_based(text)
            assert result == SOAPCategory.PLAN
    
    def test_categorize_text_rule_based_no_match(self, categorizer_default):
        """Test rule-based categorization with no pattern match"""
        text = "Some random text with no medical context"
        result = categorizer_default.categorize_text_rule_based(text)
        assert result is None
    
    def test_categorize_entity_by_type_symptom(self, categorizer_default):
        """Test entity categorization by type - symptom"""
        entity = MedicalEntity(
            text="chest pain",
            entity_type=EntityType.SYMPTOM,
            start_pos=0,
            end_pos=10,
            confidence=0.9
        )
        
        result = categorizer_default.categorize_entity_by_type(entity)
        assert result == SOAPCategory.SUBJECTIVE
    
    def test_categorize_entity_by_type_vital_sign(self, categorizer_default):
        """Test entity categorization by type - vital sign"""
        entity = MedicalEntity(
            text="blood pressure",
            entity_type=EntityType.VITAL_SIGN,
            start_pos=0,
            end_pos=14,
            confidence=0.9
        )
        
        result = categorizer_default.categorize_entity_by_type(entity)
        assert result == SOAPCategory.OBJECTIVE
    
    def test_categorize_entity_by_type_disease(self, categorizer_default):
        """Test entity categorization by type - disease"""
        entity = MedicalEntity(
            text="hypertension",
            entity_type=EntityType.DISEASE,
            start_pos=0,
            end_pos=12,
            confidence=0.9
        )
        
        result = categorizer_default.categorize_entity_by_type(entity)
        assert result == SOAPCategory.ASSESSMENT
    
    def test_categorize_entity_by_type_medication(self, categorizer_default):
        """Test entity categorization by type - medication"""
        entity = MedicalEntity(
            text="aspirin",
            entity_type=EntityType.MEDICATION,
            start_pos=0,
            end_pos=7,
            confidence=0.9
        )
        
        result = categorizer_default.categorize_entity_by_type(entity)
        assert result == SOAPCategory.PLAN
    
    def test_categorize_entity_by_type_unknown(self, categorizer_default):
        """Test entity categorization with unknown type"""
        entity = MedicalEntity(
            text="unknown entity",
            entity_type=EntityType.ANATOMY,  # Not in default mapping
            start_pos=0,
            end_pos=14,
            confidence=0.9
        )
        
        result = categorizer_default.categorize_entity_by_type(entity)
        assert result is None
    
    def test_categorize_entities_hybrid_llm_priority(self, categorizer_with_mock_client):
        """Test hybrid categorization prioritizes LLM results"""
        entities = [
            MedicalEntity(
                text="chest pain",
                entity_type=EntityType.SYMPTOM,
                start_pos=0,
                end_pos=10,
                confidence=0.9
            )
        ]
        
        text = "Patient complains of chest pain"
        
        # Mock LLM response
        categorizer_with_mock_client.client.categorize_soap.return_value = {
            "category": "SUBJECTIVE",
            "confidence": 0.95
        }
        
        result = categorizer_with_mock_client.categorize_entities_hybrid(entities, text)
        
        assert isinstance(result, dict)
        assert SOAPCategory.SUBJECTIVE in result
        assert len(result[SOAPCategory.SUBJECTIVE]) == 1
        assert result[SOAPCategory.SUBJECTIVE][0] == entities[0]
    
    def test_categorize_entities_hybrid_fallback_to_rules(self, categorizer_with_mock_client):
        """Test hybrid categorization falls back to rules when LLM fails"""
        entities = [
            MedicalEntity(
                text="chest pain",
                entity_type=EntityType.SYMPTOM,
                start_pos=0,
                end_pos=10,
                confidence=0.9
            )
        ]
        
        text = "Patient complains of chest pain"
        
        # Mock LLM failure
        categorizer_with_mock_client.client.categorize_soap.side_effect = Exception("API Error")
        
        result = categorizer_with_mock_client.categorize_entities_hybrid(entities, text)
        
        # Should fallback to entity type mapping
        assert isinstance(result, dict)
        assert SOAPCategory.SUBJECTIVE in result
        assert len(result[SOAPCategory.SUBJECTIVE]) == 1
    
    def test_categorize_clinical_note_sections(self, categorizer_default):
        """Test categorization of full clinical note sections"""
        note_sections = {
            "Chief Complaint": "Patient complains of chest pain",
            "Physical Exam": "Vital signs stable, heart rate 72 bpm",
            "Assessment": "Diagnosis: Stable angina",
            "Plan": "Start aspirin, follow up in 1 week"
        }
        
        result = categorizer_default.categorize_clinical_note_sections(note_sections)
        
        assert isinstance(result, dict)
        assert all(category in result for category in SOAPCategory)
        
        # Check that sections were categorized appropriately
        subjective_sections = result[SOAPCategory.SUBJECTIVE]
        objective_sections = result[SOAPCategory.OBJECTIVE]
        assessment_sections = result[SOAPCategory.ASSESSMENT]
        plan_sections = result[SOAPCategory.PLAN]
        
        assert len(subjective_sections) > 0
        assert len(objective_sections) > 0
        assert len(assessment_sections) > 0
        assert len(plan_sections) > 0
    
    def test_get_soap_confidence_score(self, categorizer_default):
        """Test SOAP confidence scoring"""
        # Text with strong SOAP indicators
        strong_subjective = "Patient complains of severe chest pain and reports difficulty breathing"
        strong_objective = "Vital signs: BP 140/90, HR 95, Temp 98.6F, physical exam shows rales"
        strong_assessment = "Primary diagnosis: Acute heart failure, secondary diagnosis: hypertension"
        strong_plan = "Start furosemide 40mg daily, follow up in 2 days, patient education on diet"
        
        subjective_score = categorizer_default.get_soap_confidence_score(strong_subjective, SOAPCategory.SUBJECTIVE)
        objective_score = categorizer_default.get_soap_confidence_score(strong_objective, SOAPCategory.OBJECTIVE)
        assessment_score = categorizer_default.get_soap_confidence_score(strong_assessment, SOAPCategory.ASSESSMENT)
        plan_score = categorizer_default.get_soap_confidence_score(strong_plan, SOAPCategory.PLAN)
        
        # All should have high confidence scores
        assert subjective_score > 0.5
        assert objective_score > 0.5
        assert assessment_score > 0.5
        assert plan_score > 0.5
        
        # Wrong category should have lower score
        wrong_score = categorizer_default.get_soap_confidence_score(strong_subjective, SOAPCategory.OBJECTIVE)
        assert wrong_score <= subjective_score