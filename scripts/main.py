#!/usr/bin/env python3
"""
SOAP Knowledge Graph Generator for MIMIC-IV Dataset

This script processes MIMIC-IV clinical data to construct a knowledge graph
organized by SOAP (Subjective, Objective, Assessment, Plan) categories.
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from typing import Optional

# Import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from soap_kg.config import Config
from soap_kg.core.data_loader import MimicDataLoader
from soap_kg.utils.openrouter_client import OpenRouterClient
from soap_kg.core.medical_ner import MedicalNER
from soap_kg.core.relationship_extractor import RelationshipExtractor
from soap_kg.core.soap_categorizer import SOAPCategorizer
from soap_kg.core.knowledge_graph_builder import KnowledgeGraphBuilder
from soap_kg.utils.visualization import KnowledgeGraphVisualizer
from soap_kg.models.soap_schema import SOAPNote, SOAPCategory

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Setup logging configuration"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )

def validate_environment():
    """Validate required environment variables and files"""
    errors = []
    
    # Check OpenRouter API key
    if not Config.OPENROUTER_API_KEY:
        errors.append("OPENROUTER_API_KEY not found in environment variables")
    
    # Check MIMIC-IV data path
    if not os.path.exists(Config.MIMIC_IV_PATH):
        errors.append(f"MIMIC-IV data path not found: {Config.MIMIC_IV_PATH}")
    
    if errors:
        for error in errors:
            logging.error(error)
        sys.exit(1)

def process_sample_data(n_patients: int = 10, use_llm: bool = True) -> KnowledgeGraphBuilder:
    """Process sample data and build knowledge graph"""
    logger = logging.getLogger(__name__)
    
    # Initialize components
    logger.info("Initializing components...")
    data_loader = MimicDataLoader()
    openrouter_client = OpenRouterClient() if use_llm else None
    ner = MedicalNER(openrouter_client)
    rel_extractor = RelationshipExtractor(openrouter_client)
    soap_categorizer = SOAPCategorizer(openrouter_client)
    kg_builder = KnowledgeGraphBuilder()
    
    # Load sample patient data
    logger.info(f"Loading sample data for {n_patients} patients...")
    sample_records = data_loader.get_sample_records(n_patients)
    
    # Process each patient
    for i, patient_data in enumerate(sample_records):
        patient_id = str(patient_data.get('demographics', [{}])[0].get('subject_id', f'unknown_{i}'))
        logger.info(f"Processing patient {patient_id} ({i+1}/{len(sample_records)})")
        
        try:
            # Extract clinical texts
            clinical_texts = []
            
            # Process prescriptions
            for prescription in patient_data.get('prescriptions', []):
                if prescription.get('drug'):
                    clinical_texts.append(f"Medication: {prescription['drug']}")
            
            # Process diagnoses (simplified)
            for diagnosis in patient_data.get('diagnoses', []):
                if diagnosis.get('icd_code'):
                    clinical_texts.append(f"Diagnosis code: {diagnosis['icd_code']}")
            
            # Combine all clinical texts for this patient
            combined_text = ". ".join(clinical_texts)
            
            if not combined_text.strip():
                logger.warning(f"No clinical text found for patient {patient_id}")
                continue
            
            # Extract entities
            logger.info(f"Extracting entities for patient {patient_id}")
            entities = ner.extract_entities(combined_text, use_llm=use_llm)
            
            if not entities:
                logger.warning(f"No entities extracted for patient {patient_id}")
                continue
            
            # Categorize entities into SOAP categories
            logger.info(f"Categorizing entities into SOAP categories for patient {patient_id}")
            categorized_entities = soap_categorizer.categorize_entities(
                combined_text, entities, use_llm=use_llm
            )
            
            # Extract relationships
            logger.info(f"Extracting relationships for patient {patient_id}")
            relationships = rel_extractor.extract_relationships(
                combined_text, categorized_entities, use_llm=use_llm
            )
            
            # Create SOAP note
            soap_structure = soap_categorizer.create_soap_structure(categorized_entities)
            soap_note = SOAPNote(
                patient_id=patient_id,
                admission_id=None,
                subjective=soap_structure[SOAPCategory.SUBJECTIVE],
                objective=soap_structure[SOAPCategory.OBJECTIVE],
                assessment=soap_structure[SOAPCategory.ASSESSMENT],
                plan=soap_structure[SOAPCategory.PLAN],
                relations=relationships,
                metadata={
                    'processed_at': datetime.now().isoformat(),
                    'source_texts_count': len(clinical_texts),
                    'use_llm': use_llm
                }
            )
            
            # Add to knowledge graph
            kg_builder.add_soap_note(soap_note)
            
            logger.info(f"Successfully processed patient {patient_id}: "
                       f"{len(categorized_entities)} entities, {len(relationships)} relationships")
            
        except Exception as e:
            logger.error(f"Error processing patient {patient_id}: {e}")
            continue
    
    return kg_builder

def generate_reports(kg_builder: KnowledgeGraphBuilder, output_dir: str):
    """Generate various reports and visualizations"""
    logger = logging.getLogger(__name__)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate statistics report
    logger.info("Generating statistics report...")
    stats = kg_builder.get_graph_statistics()
    
    with open(os.path.join(output_dir, "statistics.txt"), 'w') as f:
        f.write("SOAP Knowledge Graph Statistics\n")
        f.write("=" * 40 + "\n\n")
        
        for key, value in stats.items():
            if isinstance(value, dict):
                f.write(f"{key}:\n")
                for subkey, subvalue in value.items():
                    f.write(f"  {subkey}: {subvalue}\n")
            else:
                f.write(f"{key}: {value}\n")
        f.write("\n")
    
    # Generate visualizations
    logger.info("Generating visualizations...")
    visualizer = KnowledgeGraphVisualizer(kg_builder)
    
    try:
        # SOAP distribution plot
        visualizer.plot_soap_distribution(
            save_path=os.path.join(output_dir, "soap_distribution.png")
        )
        
        # Entity type distribution
        visualizer.plot_entity_type_distribution(
            save_path=os.path.join(output_dir, "entity_types.png")
        )
        
        # Network graph
        visualizer.plot_network_graph(
            save_path=os.path.join(output_dir, "network_graph.png")
        )
        
        # Relationship matrix
        visualizer.plot_relationship_matrix(
            save_path=os.path.join(output_dir, "relationship_matrix.png")
        )
        
        # Interactive dashboard
        visualizer.create_dashboard(
            save_path=os.path.join(output_dir, "dashboard.html")
        )
        
        # Interactive network
        visualizer.create_interactive_network(
            save_path=os.path.join(output_dir, "interactive_network.html")
        )
        
        logger.info(f"Visualizations saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error generating visualizations: {e}")
    
    # Export knowledge graph
    logger.info("Exporting knowledge graph...")
    try:
        kg_builder.export_to_formats(
            os.path.join(output_dir, "soap_knowledge_graph")
        )
        logger.info(f"Knowledge graph exported to {output_dir}")
    except Exception as e:
        logger.error(f"Error exporting knowledge graph: {e}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Generate SOAP Knowledge Graph from MIMIC-IV dataset"
    )
    
    parser.add_argument(
        "--patients", "-p", type=int, default=10,
        help="Number of patients to process (default: 10)"
    )
    
    parser.add_argument(
        "--output-dir", "-o", type=str, default="./output",
        help="Output directory for results (default: ./output)"
    )
    
    parser.add_argument(
        "--log-level", "-l", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--log-file", type=str, default=None,
        help="Log file path (default: console only)"
    )
    
    parser.add_argument(
        "--no-llm", action="store_true",
        help="Disable LLM-based processing (use rule-based only)"
    )
    
    parser.add_argument(
        "--data-path", type=str, default=None,
        help="Path to MIMIC-IV dataset (default: from config)"
    )
    
    parser.add_argument(
        "--load-existing", type=str, default=None,
        help="Load existing knowledge graph from pickle file"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting SOAP Knowledge Graph generation...")
    logger.info(f"Arguments: {vars(args)}")
    
    # Update config if data path provided
    if args.data_path:
        Config.MIMIC_IV_PATH = args.data_path
    
    # Validate environment
    if not args.no_llm:
        validate_environment()
    
    try:
        if args.load_existing:
            # Load existing knowledge graph
            logger.info(f"Loading existing knowledge graph from {args.load_existing}")
            kg_builder = KnowledgeGraphBuilder()
            kg_builder.load_from_file(args.load_existing)
        else:
            # Process data and build knowledge graph
            kg_builder = process_sample_data(
                n_patients=args.patients,
                use_llm=not args.no_llm
            )
        
        # Generate reports and visualizations
        generate_reports(kg_builder, args.output_dir)
        
        # Print summary
        stats = kg_builder.get_graph_statistics()
        logger.info("Processing completed successfully!")
        logger.info(f"Total entities: {stats.get('total_entities', 0)}")
        logger.info(f"Total relationships: {stats.get('total_relations', 0)}")
        logger.info(f"Total patients: {stats.get('total_patients', 0)}")
        logger.info(f"Results saved to: {args.output_dir}")
        
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()