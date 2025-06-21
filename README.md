# SOAP Knowledge Graph Generator for MIMIC-IV

This project constructs SOAP (Subjective-Objective-Assessment-Plan) based knowledge graphs from the MIMIC-IV clinical dataset using advanced NLP techniques and OpenRouter LLM services.

## Features

- **SOAP-Based Organization**: Automatically categorizes medical entities into SOAP categories
- **Advanced NER**: Uses OpenRouter LLMs for medical named entity recognition
- **Relationship Extraction**: Identifies relationships between medical entities
- **Interactive Visualizations**: Generate interactive dashboards and network visualizations
- **Multiple Export Formats**: Export knowledge graphs in JSON, GraphML, GEXF, and pickle formats
- **Scalable Processing**: Handles large datasets with configurable batch processing
- **Rule-Based Fallback**: Works without LLM APIs using built-in medical entity patterns
- **Professional Package Structure**: Properly organized Python package with CLI interface

## Project Structure

```
constructing_kg/
├── src/soap_kg/                    # Main Python package
│   ├── __init__.py                 # Package initialization
│   ├── config.py                   # Configuration settings
│   ├── cli.py                      # Command-line interface
│   ├── core/                       # Core processing modules
│   │   ├── __init__.py
│   │   ├── data_loader.py          # MIMIC-IV data loading utilities
│   │   ├── knowledge_graph_builder.py # Knowledge graph construction
│   │   ├── medical_ner.py          # Named entity recognition
│   │   ├── relationship_extractor.py # Relationship extraction
│   │   └── soap_categorizer.py     # SOAP categorization logic
│   ├── models/                     # Data models and schemas
│   │   ├── __init__.py
│   │   └── soap_schema.py          # SOAP data models
│   └── utils/                      # Utility modules
│       ├── __init__.py
│       ├── openrouter_client.py    # OpenRouter API client
│       ├── text_preprocessor.py    # Medical text preprocessing
│       └── visualization.py        # Visualization utilities
├── scripts/                        # Executable scripts
│   ├── main.py                     # Main orchestration script
│   └── setup_environment.sh        # Environment setup script
├── tests/                          # Test directory (for future tests)
├── docs/                           # Documentation directory
├── setup.py                        # Package setup script
├── pyproject.toml                  # Modern Python packaging configuration
├── MANIFEST.in                     # Package manifest
├── requirements.txt                # Python dependencies
├── environment.yml                 # Conda environment file
├── .gitignore                      # Git ignore rules
└── README.md                       # This file
```

## Setup

### 1. Create Conda Environment

```bash
# Create and activate environment
conda env create -f environment.yml
conda activate soap-kg

# Or run the setup script
chmod +x scripts/setup_environment.sh
./scripts/setup_environment.sh
```

### 2. Install Package

```bash
# For development (editable install)
pip install -e .

# For regular installation
pip install .

# Install with all optional dependencies
pip install -e .[all]
```

### 3. Configure OpenRouter API

1. Get an API key from [OpenRouter](https://openrouter.ai/)
2. Set your API key in the `.env` file:

```bash
# .env file
OPENROUTER_API_KEY=your_api_key_here
DEFAULT_MODEL=anthropic/claude-3-haiku
```

### 4. Prepare MIMIC-IV Dataset

1. Download MIMIC-IV dataset from [PhysioNet](https://physionet.org/content/mimiciv/)
2. Place the dataset in `./mimic-iv-3.1/` directory
3. Ensure the following structure:
   ```
   mimic-iv-3.1/
   ├── hosp/
   │   ├── admissions.csv.gz
   │   ├── patients.csv.gz
   │   ├── prescriptions.csv.gz
   │   └── ...
   └── icu/
       ├── chartevents.csv.gz
       └── ...
   ```

## Usage

### Basic Usage

```bash
# Process 10 patients and generate knowledge graph
python scripts/main.py

# Or use the CLI command (after pip install)
soap-kg --patients 10

# Process 50 patients with custom output directory
python scripts/main.py --patients 50 --output-dir ./results

# Use rule-based processing only (no LLM)
python scripts/main.py --no-llm --patients 20
```

### Advanced Usage

```bash
# Custom data path and logging
python scripts/main.py \
    --data-path /path/to/mimic-iv \
    --patients 100 \
    --output-dir ./large_analysis \
    --log-level DEBUG \
    --log-file processing.log

# Load existing knowledge graph for analysis
python scripts/main.py \
    --load-existing ./results/soap_knowledge_graph.pkl \
    --output-dir ./reanalysis

# Using the installed CLI
soap-kg --patients 100 --no-llm --output-dir ./test_output
```

### Command Line Options

- `--patients, -p`: Number of patients to process (default: 10)
- `--output-dir, -o`: Output directory for results (default: ./output)
- `--log-level, -l`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `--log-file`: Log file path (optional)
- `--no-llm`: Disable LLM processing, use rule-based only
- `--data-path`: Custom path to MIMIC-IV dataset
- `--load-existing`: Load existing knowledge graph from pickle file

## Output Files

The script generates several output files:

### Data Files
- `soap_knowledge_graph.json`: Complete knowledge graph in JSON format
- `soap_knowledge_graph.pkl`: Serialized knowledge graph for Python
- `soap_knowledge_graph_networkx.gexf`: NetworkX graph in GEXF format
- `soap_knowledge_graph_networkx.graphml`: NetworkX graph in GraphML format

### Reports
- `statistics.txt`: Comprehensive statistics about the knowledge graph

### Visualizations
- `soap_distribution.png`: SOAP category distribution
- `entity_types.png`: Entity type distribution  
- `network_graph.png`: Static network visualization
- `relationship_matrix.png`: Relationship type distribution
- `dashboard.html`: Interactive dashboard
- `interactive_network.html`: Interactive network visualization

## SOAP Categories

The system organizes medical entities into four SOAP categories:

- **Subjective**: Patient-reported symptoms, complaints, history
- **Objective**: Vital signs, lab results, physical exam findings, imaging
- **Assessment**: Diagnoses, impressions, differential diagnoses
- **Plan**: Treatments, medications, procedures, follow-up plans

## Entity Types

Supported medical entity types:

- `DISEASE`: Medical conditions and diagnoses
- `SYMPTOM`: Patient-reported symptoms
- `MEDICATION`: Drugs and pharmaceuticals
- `PROCEDURE`: Medical procedures and interventions
- `ANATOMY`: Anatomical structures and body parts
- `LAB_VALUE`: Laboratory test results
- `VITAL_SIGN`: Vital signs and measurements
- `TREATMENT`: Treatment modalities and therapies

## Relationship Types

The system identifies various medical relationships:

- `TREATS`: Treatment or medication treats condition
- `CAUSES`: Entity causes another entity
- `INDICATES`: Sign or test indicates condition
- `HAS_SYMPTOM`: Patient has symptom
- `DIAGNOSED_WITH`: Patient diagnosed with condition
- `LOCATED_IN`: Anatomical location relationships
- `MEASURED_BY`: Measurement relationships

## API Integration

### OpenRouter Models

The system supports various LLM models through OpenRouter:

- `anthropic/claude-3-haiku` (default, fast and cost-effective)
- `anthropic/claude-3-sonnet` (balanced performance)
- `openai/gpt-4o-mini` (alternative option)
- Custom models can be configured in `config.py`

### Rate Limiting

The OpenRouter client includes built-in rate limiting and error handling to ensure reliable processing of large datasets.

## Customization

### Adding New Entity Types

1. Update `EntityType` enum in `src/soap_kg/models/soap_schema.py`
2. Add patterns in `src/soap_kg/core/medical_ner.py`
3. Update SOAP mapping in `src/soap_kg/core/soap_categorizer.py`

### Adding New Relationship Types

1. Update `RelationType` enum in `src/soap_kg/models/soap_schema.py`
2. Add patterns in `src/soap_kg/core/relationship_extractor.py`
3. Update visualization colors if needed

### Custom Processing Pipeline

Create custom processing pipelines by importing and combining modules:

```python
from soap_kg.core.data_loader import MimicDataLoader
from soap_kg.core.medical_ner import MedicalNER
from soap_kg.core.knowledge_graph_builder import KnowledgeGraphBuilder
from soap_kg.utils.openrouter_client import OpenRouterClient

# Custom processing logic
loader = MimicDataLoader()
client = OpenRouterClient()
ner = MedicalNER(client)
kg_builder = KnowledgeGraphBuilder()

# Your custom pipeline here
```

### Development

For development work:

```bash
# Clone and install in development mode
git clone <repository-url>
cd constructing_kg
conda env create -f environment.yml
conda activate soap-kg
pip install -e .[dev]

# Run tests (when available)
pytest tests/

# Code formatting
black src/ scripts/

# Type checking
mypy src/
```

## Troubleshooting

### Common Issues

1. **Missing API Key**: Ensure `OPENROUTER_API_KEY` is set in environment
2. **Dataset Path**: Verify MIMIC-IV dataset is in correct location
3. **Memory Issues**: Reduce batch size or number of patients for large datasets
4. **Import Errors**: Ensure all dependencies are installed: `pip install -r requirements.txt`

### Performance Optimization

- Use `--no-llm` for faster processing without LLM costs
- Reduce `--patients` number for initial testing
- Adjust `MAX_BATCH_SIZE` in config for memory optimization

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Citation

If you use this work in your research, please cite:

```bibtex
@software{soap_kg_generator,
  title={SOAP Knowledge Graph Generator for MIMIC-IV},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/constructing_kg}
}
```

## Acknowledgments

- MIMIC-IV dataset from MIT Laboratory for Computational Physiology
- OpenRouter for LLM API access
- NetworkX for graph processing capabilities