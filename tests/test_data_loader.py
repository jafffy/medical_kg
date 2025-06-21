import pytest
import pandas as pd
import os
from unittest.mock import Mock, patch, MagicMock
from soap_kg.core.data_loader import MimicDataLoader


class TestMimicDataLoader:
    
    @pytest.fixture
    def mock_data_path(self, tmp_path):
        """Create a temporary data path for testing"""
        return str(tmp_path)
    
    @pytest.fixture
    def data_loader(self, mock_data_path):
        """Create a MimicDataLoader instance for testing"""
        return MimicDataLoader(data_path=mock_data_path)
    
    def test_init_with_custom_path(self, mock_data_path):
        """Test initialization with custom data path"""
        loader = MimicDataLoader(data_path=mock_data_path)
        assert loader.data_path == mock_data_path
        assert loader.tables == {}
    
    @patch('soap_kg.core.data_loader.Config')
    def test_init_with_default_path(self, mock_config):
        """Test initialization with default config path"""
        mock_config.MIMIC_IV_PATH = "/default/path"
        loader = MimicDataLoader()
        assert loader.data_path == "/default/path"
    
    def test_load_csv_file_not_found(self, data_loader):
        """Test load_csv when file doesn't exist"""
        result = data_loader.load_csv("nonexistent_table")
        assert isinstance(result, pd.DataFrame)
        assert result.empty
        assert "nonexistent_table" not in data_loader.tables
    
    @patch('soap_kg.core.data_loader.pd.read_csv')
    @patch('soap_kg.core.data_loader.os.path.exists')
    def test_load_csv_success(self, mock_exists, mock_read_csv, data_loader):
        """Test successful CSV loading"""
        mock_exists.return_value = True
        mock_df = pd.DataFrame({'col1': [1, 2], 'col2': ['a', 'b']})
        mock_read_csv.return_value = mock_df
        
        result = data_loader.load_csv("test_table")
        
        assert not result.empty
        assert "test_table" in data_loader.tables
        assert data_loader.tables["test_table"].equals(mock_df)
    
    @patch('soap_kg.core.data_loader.pd.read_csv')
    @patch('soap_kg.core.data_loader.os.path.exists')
    def test_load_csv_exception_handling(self, mock_exists, mock_read_csv, data_loader):
        """Test CSV loading with exception"""
        mock_exists.return_value = True
        mock_read_csv.side_effect = Exception("Read error")
        
        result = data_loader.load_csv("test_table")
        
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    
    def test_get_patient_data_empty_tables(self, data_loader):
        """Test get_patient_data with empty tables"""
        with patch.object(data_loader, 'load_csv') as mock_load:
            mock_load.return_value = pd.DataFrame()
            
            result = data_loader.get_patient_data(12345)
            
            assert isinstance(result, dict)
            assert 'demographics' in result
            assert 'admissions' in result
            assert 'diagnoses' in result
            assert 'prescriptions' in result
    
    def test_get_patient_data_with_data(self, data_loader):
        """Test get_patient_data with actual data"""
        # Mock patient data
        patients_df = pd.DataFrame({
            'subject_id': [12345, 12346],
            'gender': ['M', 'F'],
            'anchor_age': [65, 72]
        })
        
        # Mock admissions data
        admissions_df = pd.DataFrame({
            'subject_id': [12345, 12345],
            'hadm_id': [100, 101],
            'admission_type': ['EMERGENCY', 'ELECTIVE']
        })
        
        data_loader.tables = {
            'patients': patients_df,
            'admissions': admissions_df,
            'diagnoses_icd': pd.DataFrame(),
            'prescriptions': pd.DataFrame()
        }
        
        result = data_loader.get_patient_data(12345)
        
        assert len(result['demographics']) == 1
        assert result['demographics'][0]['subject_id'] == 12345
        assert len(result['admissions']) == 2
    
    def test_get_clinical_text_sources(self, data_loader):
        """Test clinical text sources configuration"""
        sources = data_loader.get_clinical_text_sources()
        
        assert isinstance(sources, list)
        assert len(sources) > 0
        
        # Check structure of sources
        for table_name, columns in sources:
            assert isinstance(table_name, str)
            assert isinstance(columns, list)
            assert len(columns) > 0
    
    def test_extract_clinical_texts_empty_tables(self, data_loader):
        """Test extract_clinical_texts with empty tables"""
        with patch.object(data_loader, 'load_csv') as mock_load:
            mock_load.return_value = pd.DataFrame()
            
            result = data_loader.extract_clinical_texts(limit=10)
            
            assert isinstance(result, list)
            assert len(result) == 0
    
    def test_extract_clinical_texts_with_data(self, data_loader):
        """Test extract_clinical_texts with sample data"""
        # Mock prescriptions data
        prescriptions_df = pd.DataFrame({
            'subject_id': [12345, 12346],
            'hadm_id': [100, 101],
            'drug': ['Aspirin', 'Ibuprofen'],
            'drug_name_generic': ['aspirin', 'ibuprofen']
        })
        
        with patch.object(data_loader, 'load_csv') as mock_load:
            mock_load.return_value = prescriptions_df
            
            result = data_loader.extract_clinical_texts(limit=10)
            
            assert isinstance(result, list)
            assert len(result) > 0
            
            # Check structure of result
            for item in result:
                assert 'subject_id' in item
                assert 'source_table' in item
                assert 'texts' in item
                assert isinstance(item['texts'], list)
    
    def test_get_icd_descriptions_empty_tables(self, data_loader):
        """Test get_icd_descriptions with empty tables"""
        with patch.object(data_loader, 'load_csv') as mock_load:
            mock_load.return_value = pd.DataFrame()
            
            result = data_loader.get_icd_descriptions()
            
            assert isinstance(result, dict)
            assert len(result) == 0
    
    def test_get_icd_descriptions_with_data(self, data_loader):
        """Test get_icd_descriptions with sample data"""
        # Mock ICD data
        diag_df = pd.DataFrame({
            'icd_code': ['A01.0', 'A01.1'],
            'long_title': ['Typhoid fever', 'Paratyphoid fever A']
        })
        
        proc_df = pd.DataFrame({
            'icd_code': ['0001', '0002'],
            'long_title': ['Procedure 1', 'Procedure 2']
        })
        
        def mock_load_csv(table_name, nrows=None):
            if table_name == 'd_icd_diagnoses':
                return diag_df
            elif table_name == 'd_icd_procedures':
                return proc_df
            return pd.DataFrame()
        
        with patch.object(data_loader, 'load_csv', side_effect=mock_load_csv):
            result = data_loader.get_icd_descriptions()
            
            assert isinstance(result, dict)
            assert len(result) == 4
            assert result['A01.0'] == 'Typhoid fever'
            assert result['0001'] == 'Procedure 1'
    
    def test_get_icd_descriptions_missing_keys(self, data_loader):
        """Test get_icd_descriptions handles missing keys gracefully"""
        # Mock ICD data with missing keys
        diag_df = pd.DataFrame({
            'icd_code': ['A01.0', None],
            'long_title': ['Typhoid fever', 'Missing code']
        })
        
        def mock_load_csv(table_name, nrows=None):
            if table_name == 'd_icd_diagnoses':
                return diag_df
            return pd.DataFrame()
        
        with patch.object(data_loader, 'load_csv', side_effect=mock_load_csv):
            result = data_loader.get_icd_descriptions()
            
            assert isinstance(result, dict)
            assert len(result) == 1  # Only valid entries should be included
            assert result['A01.0'] == 'Typhoid fever'
    
    def test_get_sample_records(self, data_loader):
        """Test get_sample_records functionality"""
        # Mock patient data
        patients_df = pd.DataFrame({
            'subject_id': [12345, 12346],
            'gender': ['M', 'F']
        })
        
        with patch.object(data_loader, 'load_csv') as mock_load:
            with patch.object(data_loader, 'get_patient_data') as mock_get_patient:
                mock_load.return_value = patients_df
                mock_get_patient.return_value = {'demographics': [], 'admissions': []}
                
                result = data_loader.get_sample_records(n_patients=2)
                
                assert isinstance(result, list)
                assert len(result) == 2
                assert mock_get_patient.call_count == 2