#!/usr/bin/env python3
"""
Rule-Based SOAP Knowledge Graph Generator

This version uses only rule-based processing without any LLM dependencies.
Perfect for reliable, fast processing without API costs or parsing issues.
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from typing import Optional

# Import our modules
from config import Config
from data_loader import MimicDataLoader
from medical_ner import MedicalNER
from relationship_extractor import RelationshipExtractor
from soap_categorizer import SOAPCategorizer
from knowledge_graph_builder import KnowledgeGraphBuilder
from visualization import KnowledgeGraphVisualizer
from soap_schema import SOAPNote, SOAPCategory

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

def process_rule_based_data(n_patients: int = 100) -> KnowledgeGraphBuilder:
    """Process data using only rule-based methods"""
    logger = logging.getLogger(__name__)
    
    # Initialize components (no OpenRouter client)
    logger.info("Initializing rule-based components...")
    data_loader = MimicDataLoader()
    ner = MedicalNER(openrouter_client=None)  # No LLM client
    rel_extractor = RelationshipExtractor(openrouter_client=None)  # No LLM client
    soap_categorizer = SOAPCategorizer(openrouter_client=None)  # No LLM client
    kg_builder = KnowledgeGraphBuilder()
    
    # Load sample patient data
    logger.info(f"Loading sample data for {n_patients} patients...")
    sample_records = data_loader.get_sample_records(n_patients)
    
    processed_count = 0
    successful_count = 0
    
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
                processed_count += 1
                continue
            
            # Extract entities using ONLY rule-based methods
            logger.debug(f"Extracting entities for patient {patient_id}")
            entities = ner.extract_entities(combined_text, use_llm=False)
            
            if not entities:
                logger.warning(f"No entities extracted for patient {patient_id}")
                processed_count += 1
                continue
            
            # Categorize entities into SOAP categories using ONLY rule-based methods
            logger.debug(f"Categorizing entities into SOAP categories for patient {patient_id}")
            categorized_entities = soap_categorizer.categorize_entities(
                combined_text, entities, use_llm=False
            )
            
            # Extract relationships using ONLY rule-based methods
            logger.debug(f"Extracting relationships for patient {patient_id}")
            relationships = rel_extractor.extract_relationships(
                combined_text, categorized_entities, use_llm=False
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
                    'processing_method': 'rule_based_only'
                }
            )
            
            # Add to knowledge graph
            kg_builder.add_soap_note(soap_note)
            successful_count += 1
            
            logger.info(f"Successfully processed patient {patient_id}: "
                       f"{len(categorized_entities)} entities, {len(relationships)} relationships")
            
        except Exception as e:
            logger.error(f"Error processing patient {patient_id}: {e}")
        
        processed_count += 1
        
        # Progress update every 25 patients
        if processed_count % 25 == 0:
            current_stats = kg_builder.get_graph_statistics()
            logger.info(f"Progress: {processed_count}/{len(sample_records)} patients, "
                       f"{successful_count} successful, "
                       f"{current_stats.get('total_entities', 0)} total entities")
    
    logger.info(f"Processing completed: {successful_count}/{processed_count} patients successful")
    return kg_builder

def generate_reports(kg_builder: KnowledgeGraphBuilder, output_dir: str):
    """Generate various reports and visualizations"""
    logger = logging.getLogger(__name__)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate statistics report
    logger.info("Generating statistics report...")
    stats = kg_builder.get_graph_statistics()
    
    with open(os.path.join(output_dir, "statistics.txt"), 'w') as f:
        f.write("SOAP Knowledge Graph Statistics (Rule-Based)\n")
        f.write("=" * 50 + "\n\n")
        
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
        description="Rule-Based SOAP Knowledge Graph Generator for MIMIC-IV dataset"
    )
    
    parser.add_argument(
        "--patients", "-p", type=int, default=100,
        help="Number of patients to process (default: 100)"
    )
    
    parser.add_argument(
        "--output-dir", "-o", type=str, default="./output_rules",
        help="Output directory for results (default: ./output_rules)"
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
        "--data-path", type=str, default=None,
        help="Path to MIMIC-IV dataset (default: from config)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Rule-Based SOAP Knowledge Graph generation...")
    logger.info(f"Arguments: {vars(args)}")
    
    # Update config if data path provided
    if args.data_path:
        Config.MIMIC_IV_PATH = args.data_path
    
    try:
        # Process data and build knowledge graph
        kg_builder = process_rule_based_data(n_patients=args.patients)
        
        # Generate reports and visualizations
        generate_reports(kg_builder, args.output_dir)
        
        # Print summary
        stats = kg_builder.get_graph_statistics()
        logger.info("Processing completed successfully!")
        logger.info(f"Total entities: {stats.get('total_entities', 0)}")
        logger.info(f"Total relationships: {stats.get('total_relations', 0)}")
        logger.info(f"Total patients: {stats.get('total_patients', 0)}")
        logger.info(f"Results saved to: {args.output_dir}")
        
        # Print entity breakdown
        soap_dist = stats.get('soap_distribution', {})
        logger.info("SOAP Distribution:")
        for category, count in soap_dist.items():
            logger.info(f"  {category}: {count} entities")
        
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()