#!/bin/bash

echo "Setting up SOAP Knowledge Graph environment..."

# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate soap-kg

# Download spaCy model for medical NER
python -m spacy download en_core_web_sm

# Model preferences
DEFAULT_MODEL=anthropic/claude-4-sonnet-20250522
ALTERNATIVE_MODEL=openai/gpt-4.1-2025-04-14
# Processing limits
MAX_BATCH_SIZE=100
MAX_TEXT_LENGTH=4000

echo "Environment setup complete!"
echo "Next steps:"
echo "1. conda activate soap-kg"
echo "2. Add your OpenRouter API key to .env file"
echo "3. Run: python main.py --help"