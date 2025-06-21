#!/usr/bin/env python3
"""
Example usage of the SOAP Knowledge Graph Generator

This script demonstrates how to use the various components
to process clinical text and build a knowledge graph.
"""

import logging
from data_loader import MimicDataLoader
from openrouter_client import OpenRouterClient
from medical_ner import MedicalNER
from relationship_extractor import RelationshipExtractor
from soap_categorizer import SOAPCategorizer
from knowledge_graph_builder import KnowledgeGraphBuilder
from visualization import KnowledgeGraphVisualizer
from soap_schema import SOAPNote, SOAPCategory

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def example_text_processing():
    """Example of processing a single clinical text"""
    
    # Sample clinical text
    clinical_text = """
    Patient presents with chest pain and shortness of breath. 
    Vital signs: BP 140/90, HR 95, Temp 98.6F. 
    Physical exam reveals crackles in lung bases. 
    Labs show elevated troponin at 2.5 ng/mL. 
    ECG shows ST elevation in leads II, III, aVF. 
    Assessment: Acute inferior MI. 
    Plan: Start aspirin 325mg, metoprolol 25mg BID, 
    arrange urgent cardiac catheterization.
    """
    
    logger.info("Processing sample clinical text...")
    
    # Initialize components (without OpenRouter for this example)
    ner = MedicalNER(openrouter_client=None)  # Use rule-based only
    rel_extractor = RelationshipExtractor(openrouter_client=None)
    soap_categorizer = SOAPCategorizer(openrouter_client=None)
    kg_builder = KnowledgeGraphBuilder()
    
    # Extract entities
    logger.info("Extracting entities...")
    entities = ner.extract_entities(clinical_text, use_llm=False)
    logger.info(f"Found {len(entities)} entities")
    
    # Categorize into SOAP
    logger.info("Categorizing entities into SOAP categories...")
    categorized_entities = soap_categorizer.categorize_entities(
        clinical_text, entities, use_llm=False
    )
    
    # Print SOAP categorization
    soap_structure = soap_categorizer.create_soap_structure(categorized_entities)
    for category, entity_list in soap_structure.items():
        logger.info(f"{category.value.upper()}: {len(entity_list)} entities")
        for entity in entity_list[:3]:  # Show first 3
            logger.info(f"  - {entity.text} ({entity.entity_type.value})")
    
    # Extract relationships
    logger.info("Extracting relationships...")
    relationships = rel_extractor.extract_relationships(
        clinical_text, categorized_entities, use_llm=False
    )
    logger.info(f"Found {len(relationships)} relationships")
    
    # Create SOAP note
    soap_note = SOAPNote(
        patient_id="example_patient",
        admission_id="example_admission",
        subjective=soap_structure[SOAPCategory.SUBJECTIVE],
        objective=soap_structure[SOAPCategory.OBJECTIVE],
        assessment=soap_structure[SOAPCategory.ASSESSMENT],
        plan=soap_structure[SOAPCategory.PLAN],
        relations=relationships
    )
    
    # Build knowledge graph
    logger.info("Building knowledge graph...")
    kg_builder.add_soap_note(soap_note)
    
    # Get statistics
    stats = kg_builder.get_graph_statistics()
    logger.info("Knowledge Graph Statistics:")
    for key, value in stats.items():
        if not isinstance(value, dict):
            logger.info(f"  {key}: {value}")
    
    return kg_builder

def example_mimic_processing():
    """Example of processing MIMIC-IV data (requires dataset)"""
    
    logger.info("Example MIMIC-IV processing...")
    
    try:
        # Initialize data loader
        data_loader = MimicDataLoader()
        
        # Load sample patient data
        logger.info("Loading sample patient data...")
        sample_records = data_loader.get_sample_records(n_patients=2)
        
        if not sample_records:
            logger.warning("No patient data found. Check MIMIC-IV dataset path.")
            return None
        
        logger.info(f"Loaded {len(sample_records)} patient records")
        
        # Process first patient
        patient_data = sample_records[0]
        patient_id = str(patient_data.get('demographics', [{}])[0].get('subject_id', 'unknown'))
        
        logger.info(f"Processing patient {patient_id}")
        logger.info(f"  Demographics: {len(patient_data.get('demographics', []))}")
        logger.info(f"  Admissions: {len(patient_data.get('admissions', []))}")
        logger.info(f"  Diagnoses: {len(patient_data.get('diagnoses', []))}")
        logger.info(f"  Prescriptions: {len(patient_data.get('prescriptions', []))}")
        
        return patient_data
        
    except Exception as e:
        logger.error(f"Error processing MIMIC-IV data: {e}")
        logger.info("This is expected if MIMIC-IV dataset is not available")
        return None

def example_visualization():
    """Example of creating visualizations"""
    
    logger.info("Creating example visualization...")
    
    # Build a small knowledge graph
    kg_builder = example_text_processing()
    
    # Create visualizer
    visualizer = KnowledgeGraphVisualizer(kg_builder)
    
    try:
        # Generate plots (save to current directory)
        logger.info("Generating SOAP distribution plot...")
        visualizer.plot_soap_distribution(save_path="example_soap_dist.png")
        
        logger.info("Generating network graph...")
        visualizer.plot_network_graph(save_path="example_network.png")
        
        logger.info("Creating interactive dashboard...")
        visualizer.create_dashboard(save_path="example_dashboard.html")
        
        logger.info("Visualizations created successfully!")
        
    except Exception as e:
        logger.error(f"Error creating visualizations: {e}")
        logger.info("This might be due to missing display or GUI backend")

def main():
    """Run all examples"""
    
    logger.info("=== SOAP Knowledge Graph Generator Examples ===")
    
    # Example 1: Text processing
    logger.info("\n1. Text Processing Example")
    logger.info("-" * 30)
    kg_builder = example_text_processing()
    
    # Example 2: MIMIC-IV processing
    logger.info("\n2. MIMIC-IV Data Example")
    logger.info("-" * 30)
    patient_data = example_mimic_processing()
    
    # Example 3: Visualization
    logger.info("\n3. Visualization Example")
    logger.info("-" * 30)
    example_visualization()
    
    logger.info("\n=== Examples Complete ===")
    logger.info("Check the generated files:")
    logger.info("  - example_soap_dist.png")
    logger.info("  - example_network.png") 
    logger.info("  - example_dashboard.html")

if __name__ == "__main__":
    main()