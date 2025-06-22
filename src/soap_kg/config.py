import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Data and API Configuration
    MIMIC_IV_PATH = "./mimic-iv-3.1"
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    
    # Model configurations
    DEFAULT_MODEL = "anthropic/claude-3-haiku"
    ALTERNATIVE_MODELS = [
        "openai/gpt-4o-mini",
        "meta-llama/llama-3.1-8b-instruct:free"
    ]
    
    # SOAP categories
    SOAP_CATEGORIES = {
        "subjective": ["symptom", "complaint", "history", "patient_report"],
        "objective": ["vital_signs", "lab_results", "physical_exam", "imaging"],
        "assessment": ["diagnosis", "differential", "impression", "evaluation"],
        "plan": ["treatment", "medication", "procedure", "follow_up", "discharge"]
    }
    
    # Medical entity types
    ENTITY_TYPES = [
        "DISEASE", "SYMPTOM", "MEDICATION", "PROCEDURE", 
        "ANATOMY", "LAB_VALUE", "VITAL_SIGN", "TREATMENT"
    ]
    
    # Graph configuration
    MAX_NODES = 50000
    MAX_EDGES = 100000
    
    # Processing configuration
    BATCH_SIZE = 10
    MAX_TEXT_LENGTH = 10000
    CHECKPOINT_INTERVAL = 50
    
    # API Management
    RATE_LIMIT_REQUESTS_PER_MINUTE = 60
    RATE_LIMIT_TOKENS_PER_MINUTE = 100000
    REQUEST_TIMEOUT = 60
    MAX_RETRIES = 3
    RETRY_BACKOFF_FACTOR = 2
    
    # Error Handling
    MAX_PATIENT_PROCESSING_RETRIES = 2
    SKIP_PATIENT_ON_CRITICAL_ERROR = True
    LOG_FAILED_PATIENTS = True
    FALLBACK_TO_RULES_ON_API_FAILURE = True
    
    # Quality Control
    MIN_ENTITIES_PER_PATIENT = 1
    MIN_CONFIDENCE_THRESHOLD = 0.3
    ENABLE_ENTITY_VALIDATION = True
    
    # Security Configuration
    ENABLE_API_KEY_VALIDATION = True
    MASK_SENSITIVE_DATA_IN_LOGS = True
    MIN_API_KEY_LENGTH = 20
    ALLOW_TEST_API_KEYS = False  # Set to True in development environments
    
    # SSL and Request Security
    VERIFY_SSL_CERTIFICATES = True
    MAX_REQUEST_SIZE_BYTES = 1048576  # 1MB
    MAX_RESPONSE_SIZE_BYTES = 2097152  # 2MB
    SANITIZE_INPUT_TEXT = True
    BLOCK_SUSPICIOUS_PATTERNS = True
    
    # Content Security
    ALLOWED_CONTENT_TYPES = ["application/json"]
    MAX_PROMPT_LENGTH = 50000
    ENABLE_REQUEST_LOGGING = True
    LOG_SECURITY_EVENTS = True
    
    # Performance Configuration
    MAX_ENTITIES_FOR_RELATIONSHIP_EXTRACTION = 100
    MAX_ENTITY_PAIRS_PER_BATCH = 1000
    ENABLE_ENTITY_INDEXING = True
    ENABLE_PARALLEL_PROCESSING = False  # Set to True for large datasets
    RELATIONSHIP_WINDOW_SIZE = 50
    MAX_RELATIONSHIP_PATTERNS_PER_TYPE = 5
    ENABLE_PERFORMANCE_MONITORING = True