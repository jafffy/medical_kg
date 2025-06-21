"""
Pytest configuration and shared fixtures for SOAP Knowledge Graph tests.
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch
from soap_kg.models.soap_schema import MedicalEntity, EntityType, SOAPCategory, MedicalRelation, RelationType


@pytest.fixture
def sample_medical_entities():
    """Create sample medical entities for testing"""
    return [
        MedicalEntity(
            id="ent_1",
            text="hypertension",
            entity_type=EntityType.DISEASE,
            soap_category=SOAPCategory.ASSESSMENT,
            confidence=0.95
        ),
        MedicalEntity(
            id="ent_2",
            text="aspirin",
            entity_type=EntityType.MEDICATION,
            soap_category=SOAPCategory.PLAN,
            confidence=0.90
        ),
        MedicalEntity(
            id="ent_3",
            text="chest pain",
            entity_type=EntityType.SYMPTOM,
            soap_category=SOAPCategory.SUBJECTIVE,
            confidence=0.85
        ),
        MedicalEntity(
            id="ent_4",
            text="blood pressure",
            entity_type=EntityType.VITAL_SIGN,
            soap_category=SOAPCategory.OBJECTIVE,
            confidence=0.88
        ),
        MedicalEntity(
            id="ent_5",
            text="glucose",
            entity_type=EntityType.LAB_VALUE,
            soap_category=SOAPCategory.OBJECTIVE,
            confidence=0.92
        )
    ]


@pytest.fixture
def sample_relationships(sample_medical_entities):
    """Create sample relationships for testing"""
    return [
        MedicalRelation(
            id="rel_1",
            source_entity="ent_2",  # aspirin
            target_entity="ent_1",  # hypertension
            relation_type=RelationType.TREATS,
            confidence=0.90,
            soap_context=SOAPCategory.PLAN
        ),
        MedicalRelation(
            id="rel_2",
            source_entity="ent_3",  # chest pain
            target_entity="ent_1",  # hypertension
            relation_type=RelationType.INDICATES,
            confidence=0.75,
            soap_context=SOAPCategory.ASSESSMENT
        )
    ]


@pytest.fixture
def sample_clinical_text():
    """Sample clinical text for testing"""
    return """
    SUBJECTIVE: Patient complains of chest pain and shortness of breath. 
    Reports feeling dizzy and nauseous. Family history of heart disease.
    
    OBJECTIVE: Vital signs: BP 150/95, HR 85, Temp 98.6F, RR 18. 
    Physical exam reveals irregular heart rhythm. 
    Lab results show elevated troponin levels.
    
    ASSESSMENT: Primary diagnosis is acute coronary syndrome. 
    Secondary diagnosis includes hypertension and anxiety.
    
    PLAN: Start aspirin 81mg daily and metoprolol 25mg BID. 
    Schedule cardiac catheterization. Patient education on diet and exercise.
    Follow-up in 1 week.
    """


@pytest.fixture
def sample_mimic_data():
    """Create sample MIMIC-IV data for testing"""
    return {
        'patients': pd.DataFrame({
            'subject_id': [10000032, 10000033, 10000034],
            'gender': ['M', 'F', 'M'],
            'anchor_age': [65, 72, 58],
            'anchor_year': [2180, 2119, 2178]
        }),
        'admissions': pd.DataFrame({
            'subject_id': [10000032, 10000033, 10000034],
            'hadm_id': [20000001, 20000002, 20000003],
            'admission_type': ['EMERGENCY', 'ELECTIVE', 'URGENT'],
            'admission_location': ['EMERGENCY ROOM', 'CLINIC', 'TRANSFER']
        }),
        'diagnoses_icd': pd.DataFrame({
            'subject_id': [10000032, 10000033, 10000034],
            'hadm_id': [20000001, 20000002, 20000003],
            'icd_code': ['I10', 'E11.9', 'J44.1'],
            'icd_version': [10, 10, 10]
        }),
        'prescriptions': pd.DataFrame({
            'subject_id': [10000032, 10000033, 10000034],
            'hadm_id': [20000001, 20000002, 20000003],
            'drug': ['Aspirin', 'Metformin', 'Albuterol'],
            'drug_name_generic': ['aspirin', 'metformin', 'albuterol']
        }),
        'd_icd_diagnoses': pd.DataFrame({
            'icd_code': ['I10', 'E11.9', 'J44.1'],
            'long_title': ['Essential hypertension', 'Type 2 diabetes mellitus', 'COPD with acute exacerbation']
        }),
        'd_icd_procedures': pd.DataFrame({
            'icd_code': ['0001', '0002', '0003'],
            'long_title': ['Cardiac catheterization', 'Blood transfusion', 'Mechanical ventilation']
        })
    }


@pytest.fixture
def mock_openrouter_responses():
    """Mock responses from OpenRouter API"""
    return {
        'extract_entities': [
            {
                "text": "hypertension",
                "type": "DISEASE",
                "confidence": 0.95,
                "start": 12,
                "end": 24
            },
            {
                "text": "aspirin",
                "type": "MEDICATION",
                "confidence": 0.90,
                "start": 35,
                "end": 42
            }
        ],
        'extract_relationships': [
            {
                "source_entity": "aspirin",
                "target_entity": "hypertension",
                "relationship_type": "TREATS",
                "confidence": 0.88
            }
        ],
        'categorize_soap': {
            "category": "SUBJECTIVE",
            "confidence": 0.92,
            "reasoning": "Patient is describing symptoms"
        }
    }


@pytest.fixture
def temp_mimic_data_files(tmp_path, sample_mimic_data):
    """Create temporary MIMIC data files for testing"""
    data_dir = tmp_path / "mimic_data"
    hosp_dir = data_dir / "hosp"
    icu_dir = data_dir / "icu"
    
    hosp_dir.mkdir(parents=True)
    icu_dir.mkdir(parents=True)
    
    # Save sample data as CSV files
    for table_name, df in sample_mimic_data.items():
        file_path = hosp_dir / f"{table_name}.csv"
        df.to_csv(file_path, index=False)
    
    return str(data_dir)


# Global test configuration
@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment with necessary patches"""
    with patch('soap_kg.config.Config.MIMIC_IV_PATH', '/tmp/test_mimic'):
        with patch('soap_kg.utils.openrouter_client.OpenRouterClient'):
            yield


class MockOpenRouterClient:
    """Mock OpenRouter client for testing"""
    
    def __init__(self, responses=None):
        self.responses = responses or {}
        
    def extract_entities(self, text):
        if 'extract_entities' in self.responses:
            return self.responses['extract_entities']
        return []
    
    def extract_relationships(self, entities, text):
        if 'extract_relationships' in self.responses:
            return self.responses['extract_relationships']
        return []
    
    def categorize_soap(self, text):
        if 'categorize_soap' in self.responses:
            return self.responses['categorize_soap']
        return None


@pytest.fixture
def mock_openrouter_client(mock_openrouter_responses):
    """Create a mock OpenRouter client with predefined responses"""
    return MockOpenRouterClient(mock_openrouter_responses)