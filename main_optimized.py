#!/usr/bin/env python3
"""
Optimized SOAP Knowledge Graph Generator for Large-Scale MIMIC-IV Processing

This version includes improvements for processing large numbers of patients:
- Better error handling and recovery
- Rate limiting and API management
- Batch processing optimizations
- Progress checkpointing
"""

import argparse
import logging
import os
import sys
import time
import pickle
from datetime import datetime
from typing import Optional

# Import our modules
from config import Config
from data_loader import MimicDataLoader
from openrouter_client import OpenRouterClient
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

def save_checkpoint(kg_builder: KnowledgeGraphBuilder, checkpoint_file: str, processed_count: int):
    """Save processing checkpoint"""
    checkpoint_data = {
        'kg_builder': kg_builder,
        'processed_count': processed_count,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    
    logging.info(f"Checkpoint saved: {processed_count} patients processed")

def load_checkpoint(checkpoint_file: str) -> tuple:
    """Load processing checkpoint"""
    try:
        with open(checkpoint_file, 'rb') as f:
            checkpoint_data = pickle.load(f)
        
        kg_builder = checkpoint_data['kg_builder']
        processed_count = checkpoint_data['processed_count']
        
        logging.info(f"Checkpoint loaded: resuming from {processed_count} patients")
        return kg_builder, processed_count
    
    except FileNotFoundError:
        logging.info("No checkpoint found, starting fresh")
        return None, 0
    except Exception as e:
        logging.error(f"Error loading checkpoint: {e}")
        return None, 0

def process_patient_with_retry(patient_data, patient_id, ner, rel_extractor, 
                             soap_categorizer, use_llm=True, max_retries=2):
    """Process a single patient with retry logic"""
    
    for attempt in range(max_retries + 1):
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
                return None, "No clinical text found"
            
            # Extract entities with rate limiting
            if use_llm and attempt > 0:
                time.sleep(1)  # Rate limiting between retries
            
            entities = ner.extract_entities(combined_text, use_llm=use_llm)
            
            if not entities:
                return None, "No entities extracted"
            
            # Categorize entities into SOAP categories
            categorized_entities = soap_categorizer.categorize_entities(
                combined_text, entities, use_llm=use_llm
            )
            
            # Extract relationships
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
                    'use_llm': use_llm,
                    'attempt': attempt + 1
                }
            )
            
            return soap_note, None
            
        except Exception as e:
            error_msg = f"Attempt {attempt + 1} failed: {e}"
            logging.warning(f"Error processing patient {patient_id}: {error_msg}")
            
            if attempt < max_retries:
                wait_time = (attempt + 1) * 2  # Progressive backoff
                logging.info(f"Retrying patient {patient_id} in {wait_time}s...")
                time.sleep(wait_time)
                continue
            else:
                return None, error_msg
    
    return None, "Max retries exceeded"

def process_large_dataset(n_patients: int = 1000, use_llm: bool = True, 
                         checkpoint_interval: int = 50, output_dir: str = "./output") -> KnowledgeGraphBuilder:
    """Process large dataset with optimizations"""
    logger = logging.getLogger(__name__)
    
    # Setup checkpoint file
    checkpoint_file = os.path.join(output_dir, f"checkpoint_{n_patients}.pkl")
    os.makedirs(output_dir, exist_ok=True)
    
    # Try to load existing checkpoint
    kg_builder, start_index = load_checkpoint(checkpoint_file)
    
    if kg_builder is None:
        # Initialize components
        logger.info("Initializing components...")
        data_loader = MimicDataLoader()
        openrouter_client = OpenRouterClient() if use_llm else None
        ner = MedicalNER(openrouter_client)
        rel_extractor = RelationshipExtractor(openrouter_client)
        soap_categorizer = SOAPCategorizer(openrouter_client)
        kg_builder = KnowledgeGraphBuilder()
    else:
        # Reinitialize processing components (don't serialize these)
        logger.info("Reinitializing processing components...")
        data_loader = MimicDataLoader()
        openrouter_client = OpenRouterClient() if use_llm else None
        ner = MedicalNER(openrouter_client)
        rel_extractor = RelationshipExtractor(openrouter_client)
        soap_categorizer = SOAPCategorizer(openrouter_client)
    
    # Load patient data
    logger.info(f"Loading sample data for {n_patients} patients...")
    sample_records = data_loader.get_sample_records(n_patients)
    
    # Process each patient
    processed_count = start_index
    successful_count = 0
    
    for i in range(start_index, min(len(sample_records), n_patients)):
        patient_data = sample_records[i]
        patient_id = str(patient_data.get('demographics', [{}])[0].get('subject_id', f'unknown_{i}'))
        
        logger.info(f"Processing patient {patient_id} ({i+1}/{n_patients})")
        
        # Process patient with retry logic
        soap_note, error = process_patient_with_retry(
            patient_data, patient_id, ner, rel_extractor, soap_categorizer, use_llm
        )
        
        if soap_note:
            kg_builder.add_soap_note(soap_note)
            successful_count += 1
            
            entities_count = len(soap_note.get_all_entities())
            relations_count = len(soap_note.relations)
            logger.info(f"Successfully processed patient {patient_id}: "
                       f"{entities_count} entities, {relations_count} relationships")
        else:
            logger.warning(f"Failed to process patient {patient_id}: {error}")
        
        processed_count += 1
        
        # Save checkpoint periodically
        if processed_count % checkpoint_interval == 0:
            save_checkpoint(kg_builder, checkpoint_file, processed_count)
            
            # Log progress
            current_stats = kg_builder.get_graph_statistics()
            logger.info(f"Progress: {processed_count}/{n_patients} patients, "
                       f"{successful_count} successful, "
                       f"{current_stats.get('total_entities', 0)} total entities")
        
        # Rate limiting for API calls
        if use_llm and i % 10 == 0:
            time.sleep(1)  # Brief pause every 10 patients
    
    # Final checkpoint
    save_checkpoint(kg_builder, checkpoint_file, processed_count)
    
    logger.info(f"Processing completed: {successful_count}/{processed_count} patients successful")
    return kg_builder

def generate_reports_optimized(kg_builder: KnowledgeGraphBuilder, output_dir: str):
    """Generate reports optimized for large datasets"""
    logger = logging.getLogger(__name__)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate statistics report
    logger.info("Generating statistics report...")
    stats = kg_builder.get_graph_statistics()
    
    with open(os.path.join(output_dir, "statistics.txt"), 'w') as f:
        f.write("SOAP Knowledge Graph Statistics (Large Scale)\n")
        f.write("=" * 50 + "\n\n")
        
        for key, value in stats.items():
            if isinstance(value, dict):
                f.write(f"{key}:\n")
                for subkey, subvalue in value.items():
                    f.write(f"  {subkey}: {subvalue}\n")
            else:
                f.write(f"{key}: {value}\n")
        f.write("\n")
    
    # Generate visualizations (with sampling for large graphs)
    logger.info("Generating visualizations...")
    visualizer = KnowledgeGraphVisualizer(kg_builder)
    
    try:
        # Basic distribution plots (fast)
        visualizer.plot_soap_distribution(
            save_path=os.path.join(output_dir, "soap_distribution.png")
        )
        
        visualizer.plot_entity_type_distribution(
            save_path=os.path.join(output_dir, "entity_types.png")
        )
        
        visualizer.plot_relationship_matrix(
            save_path=os.path.join(output_dir, "relationship_matrix.png")
        )
        
        # Sample network graph for large datasets
        max_nodes = min(200, kg_builder.networkx_graph.number_of_nodes())
        visualizer.plot_network_graph(
            max_nodes=max_nodes,
            save_path=os.path.join(output_dir, "network_graph_sample.png")
        )
        
        # Interactive dashboard (sampled)
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
    """Main function for optimized processing"""
    parser = argparse.ArgumentParser(
        description="Optimized SOAP Knowledge Graph Generator for large-scale MIMIC-IV processing"
    )
    
    parser.add_argument(
        "--patients", "-p", type=int, default=1000,
        help="Number of patients to process (default: 1000)"
    )
    
    parser.add_argument(
        "--output-dir", "-o", type=str, default="./output_large",
        help="Output directory for results (default: ./output_large)"
    )
    
    parser.add_argument(
        "--checkpoint-interval", "-c", type=int, default=50,
        help="Save checkpoint every N patients (default: 50)"
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
        "--resume", action="store_true",
        help="Resume from checkpoint if available"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting optimized SOAP Knowledge Graph generation...")
    logger.info(f"Arguments: {vars(args)}")
    
    try:
        # Process data and build knowledge graph
        kg_builder = process_large_dataset(
            n_patients=args.patients,
            use_llm=not args.no_llm,
            checkpoint_interval=args.checkpoint_interval,
            output_dir=args.output_dir
        )
        
        # Generate reports and visualizations
        generate_reports_optimized(kg_builder, args.output_dir)
        
        # Print summary
        stats = kg_builder.get_graph_statistics()
        logger.info("Processing completed successfully!")
        logger.info(f"Total entities: {stats.get('total_entities', 0)}")
        logger.info(f"Total relationships: {stats.get('total_relations', 0)}")
        logger.info(f"Total patients: {stats.get('total_patients', 0)}")
        logger.info(f"Results saved to: {args.output_dir}")
        
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        logger.info("Progress has been saved in checkpoint file")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()