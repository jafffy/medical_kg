import pytest
from unittest.mock import Mock, patch, mock_open
from soap_kg.utils.text_preprocessor import MedicalTextPreprocessor
from soap_kg.utils.openrouter_client import OpenRouterClient


class TestMedicalTextPreprocessor:
    
    @pytest.fixture
    def preprocessor(self):
        """Create a MedicalTextPreprocessor instance"""
        return MedicalTextPreprocessor()
    
    def test_init(self, preprocessor):
        """Test preprocessor initialization"""
        assert hasattr(preprocessor, 'medical_abbreviations')
        assert hasattr(preprocessor, 'stopwords')
        assert isinstance(preprocessor.medical_abbreviations, dict)
        assert isinstance(preprocessor.stopwords, set)
    
    def test_expand_abbreviations(self, preprocessor):
        """Test medical abbreviation expansion"""
        text = "Patient has HTN and DM"
        result = preprocessor.expand_abbreviations(text)
        
        # Should expand common medical abbreviations
        assert "HTN" not in result or "hypertension" in result.lower()
        assert "DM" not in result or "diabetes" in result.lower()
    
    def test_clean_text_basic(self, preprocessor):
        """Test basic text cleaning"""
        text = "  Patient   has\n\nmultiple    spaces\t\tand\nnewlines  "
        result = preprocessor.clean_text(text)
        
        # Should normalize whitespace
        assert "  " not in result
        assert "\n\n" not in result
        assert "\t\t" not in result
        assert result.strip() == result
    
    def test_clean_text_special_characters(self, preprocessor):
        """Test special character handling"""
        text = "Patient's BP: 120/80 mmHg (normal)"
        result = preprocessor.clean_text(text)
        
        # Should preserve important medical characters
        assert "120/80" in result
        assert ":" in result or result.find("BP") < result.find("120")
    
    def test_tokenize_medical_text(self, preprocessor):
        """Test medical text tokenization"""
        text = "Patient diagnosed with hypertension and diabetes mellitus"
        tokens = preprocessor.tokenize_medical_text(text)
        
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert "hypertension" in tokens
        assert "diabetes" in tokens
    
    def test_preprocess_pipeline(self, preprocessor):
        """Test complete preprocessing pipeline"""
        text = "Pt c/o SOB and CP. HTN dx'd."
        result = preprocessor.preprocess(text)
        
        assert isinstance(result, str)
        assert len(result) > 0
        # Should be cleaner than original
        assert len(result.split()) >= len(text.split())


class TestOpenRouterClient:
    
    @pytest.fixture
    def mock_env_vars(self):
        """Mock environment variables"""
        with patch.dict('os.environ', {
            'OPENROUTER_API_KEY': 'test_api_key',
            'OPENROUTER_MODEL': 'test_model'
        }):
            yield
    
    @pytest.fixture
    @patch('soap_kg.utils.openrouter_client.requests')
    def client(self, mock_requests, mock_env_vars):
        """Create OpenRouterClient with mocked requests"""
        return OpenRouterClient()
    
    def test_init_with_env_vars(self, mock_env_vars):
        """Test initialization with environment variables"""
        with patch('soap_kg.utils.openrouter_client.requests'):
            client = OpenRouterClient()
            assert client.api_key == 'test_api_key'
            assert client.model == 'test_model'
    
    def test_init_with_parameters(self):
        """Test initialization with explicit parameters"""
        with patch('soap_kg.utils.openrouter_client.requests'):
            client = OpenRouterClient(api_key='custom_key', model='custom_model')
            assert client.api_key == 'custom_key'
            assert client.model == 'custom_model'
    
    @patch('soap_kg.utils.openrouter_client.requests.post')
    def test_make_request_success(self, mock_post, client):
        """Test successful API request"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'result': 'success'}
        mock_post.return_value = mock_response
        
        result = client._make_request({'test': 'data'})
        
        assert result == {'result': 'success'}
        mock_post.assert_called_once()
    
    @patch('soap_kg.utils.openrouter_client.requests.post')
    def test_make_request_failure(self, mock_post, client):
        """Test failed API request"""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = 'Bad Request'
        mock_post.return_value = mock_response
        
        with pytest.raises(Exception):
            client._make_request({'test': 'data'})
    
    @patch('soap_kg.utils.openrouter_client.requests.post')
    def test_make_request_network_error(self, mock_post, client):
        """Test network error handling"""
        mock_post.side_effect = Exception("Network error")
        
        with pytest.raises(Exception):
            client._make_request({'test': 'data'})
    
    def test_extract_entities_input_validation(self, client):
        """Test entity extraction input validation"""
        # Empty text
        result = client.extract_medical_entities("")
        assert result == []
        
        # None text
        result = client.extract_medical_entities(None)
        assert result == []
    
    @patch.object(OpenRouterClient, '_make_request')
    def test_extract_entities_success(self, mock_request, client):
        """Test successful entity extraction"""
        mock_request.return_value = {
            'choices': [{
                'message': {
                    'content': '[{"text": "hypertension", "type": "DISEASE", "confidence": 0.95}]'
                }
            }]
        }
        
        result = client.extract_entities("Patient has hypertension")
        
        assert len(result) == 1
        assert result[0]['text'] == 'hypertension'
        assert result[0]['type'] == 'DISEASE'
    
    @patch.object(OpenRouterClient, '_make_request')
    def test_extract_entities_parse_error(self, mock_request, client):
        """Test entity extraction with JSON parse error"""
        mock_request.return_value = {
            'choices': [{
                'message': {
                    'content': 'Invalid JSON response'
                }
            }]
        }
        
        result = client.extract_entities("Patient has hypertension")
        assert result == []
    
    def test_extract_relationships_input_validation(self, client):
        """Test relationship extraction input validation"""
        # Empty entities
        result = client.extract_relationships([], "Some text")
        assert result == []
        
        # Empty text
        result = client.extract_relationships([{'text': 'entity'}], "")
        assert result == []
    
    @patch.object(OpenRouterClient, '_make_request')
    def test_extract_relationships_success(self, mock_request, client):
        """Test successful relationship extraction"""
        mock_request.return_value = {
            'choices': [{
                'message': {
                    'content': '[{"source": "aspirin", "target": "hypertension", "type": "TREATS"}]'
                }
            }]
        }
        
        entities = [{'text': 'aspirin'}, {'text': 'hypertension'}]
        result = client.extract_relationships(entities, "Patient takes aspirin for hypertension")
        
        assert len(result) == 1
        assert result[0]['source'] == 'aspirin'
        assert result[0]['target'] == 'hypertension'
    
    def test_categorize_soap_input_validation(self, client):
        """Test SOAP categorization input validation"""
        # Empty text
        result = client.categorize_soap("")
        assert result is None
        
        # None text
        result = client.categorize_soap(None)
        assert result is None
    
    @patch.object(OpenRouterClient, '_make_request')
    def test_categorize_soap_success(self, mock_request, client):
        """Test successful SOAP categorization"""
        mock_request.return_value = {
            'choices': [{
                'message': {
                    'content': '{"category": "SUBJECTIVE", "confidence": 0.92}'
                }
            }]
        }
        
        result = client.categorize_soap("Patient complains of chest pain")
        
        assert result['category'] == 'SUBJECTIVE'
        assert result['confidence'] == 0.92
    
    def test_build_entity_prompt(self, client):
        """Test entity extraction prompt building"""
        text = "Patient has hypertension and takes aspirin"
        prompt = client._build_entity_prompt(text)
        
        assert isinstance(prompt, str)
        assert text in prompt
        assert "extract" in prompt.lower()
        assert "entities" in prompt.lower()
    
    def test_build_relationship_prompt(self, client):
        """Test relationship extraction prompt building"""
        entities = [{'text': 'aspirin'}, {'text': 'hypertension'}]
        text = "Patient takes aspirin for hypertension"
        prompt = client._build_relationship_prompt(entities, text)
        
        assert isinstance(prompt, str)
        assert text in prompt
        assert "aspirin" in prompt
        assert "hypertension" in prompt
        assert "relationship" in prompt.lower()
    
    def test_build_soap_prompt(self, client):
        """Test SOAP categorization prompt building"""
        text = "Patient complains of chest pain"
        prompt = client._build_soap_prompt(text)
        
        assert isinstance(prompt, str)
        assert text in prompt
        assert "SOAP" in prompt
        assert any(category in prompt for category in ["SUBJECTIVE", "OBJECTIVE", "ASSESSMENT", "PLAN"])
    
    def test_rate_limiting(self, client):
        """Test rate limiting functionality"""
        # This is a basic test - in practice, you'd need to test actual timing
        with patch('time.sleep') as mock_sleep:
            client._apply_rate_limit()
            # Should not raise an error
            assert True
    
    @patch.object(OpenRouterClient, '_make_request')
    def test_retry_mechanism(self, mock_request, client):
        """Test retry mechanism for failed requests"""
        # First call fails, second succeeds
        mock_request.side_effect = [
            Exception("Temporary failure"),
            {'choices': [{'message': {'content': '[]'}}]}
        ]
        
        # Should not raise exception due to retry
        result = client.extract_entities("test text")
        assert result == []
        assert mock_request.call_count == 2