import pandas as pd
import gzip
import os
from typing import Dict, List, Optional, Tuple
from soap_kg.config import Config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MimicDataLoader:
    def __init__(self, data_path: str = None):
        self.data_path = data_path or Config.MIMIC_IV_PATH
        self.tables = {}
        
    def load_csv(self, table_name: str, nrows: Optional[int] = None) -> pd.DataFrame:
        """Load a specific CSV table from MIMIC-IV dataset"""
        file_paths = [
            os.path.join(self.data_path, "hosp", f"{table_name}.csv.gz"),
            os.path.join(self.data_path, "icu", f"{table_name}.csv.gz")
        ]
        
        for file_path in file_paths:
            if os.path.exists(file_path):
                logger.info(f"Loading {table_name} from {file_path}")
                try:
                    df = pd.read_csv(file_path, nrows=nrows)
                    self.tables[table_name] = df
                    return df
                except Exception as e:
                    logger.error(f"Error loading {table_name}: {e}")
                    return pd.DataFrame()
        
        logger.warning(f"Table {table_name} not found in {self.data_path}")
        return pd.DataFrame()
    
    def get_patient_data(self, subject_id: int) -> Dict:
        """Get comprehensive data for a specific patient"""
        patient_data = {}
        
        # Load basic patient info
        if 'patients' not in self.tables:
            self.load_csv('patients')
        
        patient_info = self.tables.get('patients', pd.DataFrame())
        if not patient_info.empty:
            patient_data['demographics'] = patient_info[
                patient_info['subject_id'] == subject_id
            ].to_dict('records')
        
        # Load admissions
        if 'admissions' not in self.tables:
            self.load_csv('admissions')
        
        admissions = self.tables.get('admissions', pd.DataFrame())
        if not admissions.empty:
            patient_data['admissions'] = admissions[
                admissions['subject_id'] == subject_id
            ].to_dict('records')
        
        # Load diagnoses
        if 'diagnoses_icd' not in self.tables:
            self.load_csv('diagnoses_icd')
        
        diagnoses = self.tables.get('diagnoses_icd', pd.DataFrame())
        if not diagnoses.empty:
            patient_data['diagnoses'] = diagnoses[
                diagnoses['subject_id'] == subject_id
            ].to_dict('records')
        
        # Load prescriptions
        if 'prescriptions' not in self.tables:
            self.load_csv('prescriptions')
        
        prescriptions = self.tables.get('prescriptions', pd.DataFrame())
        if not prescriptions.empty:
            patient_data['prescriptions'] = prescriptions[
                prescriptions['subject_id'] == subject_id
            ].to_dict('records')
        
        return patient_data
    
    def get_clinical_text_sources(self) -> List[Tuple[str, List[str]]]:
        """Identify tables and columns containing clinical text"""
        text_sources = [
            ('prescriptions', ['drug', 'drug_name_generic', 'formulary_drug_cd']),
            ('diagnoses_icd', ['icd_code']),
            ('procedures_icd', ['icd_code']),
            ('poe_detail', ['field_name', 'field_value']),
        ]
        
        return text_sources
    
    def extract_clinical_texts(self, limit: int = 1000) -> List[Dict]:
        """Extract clinical texts from various MIMIC tables"""
        clinical_texts = []
        
        for table_name, text_columns in self.get_clinical_text_sources():
            logger.info(f"Processing {table_name} for clinical text")
            
            df = self.load_csv(table_name, nrows=limit)
            if df.empty:
                continue
            
            for _, row in df.iterrows():
                text_data = {
                    'subject_id': row.get('subject_id', None),
                    'hadm_id': row.get('hadm_id', None),
                    'source_table': table_name,
                    'texts': []
                }
                
                for col in text_columns:
                    if col in df.columns and pd.notna(row[col]):
                        text_data['texts'].append({
                            'field': col,
                            'content': str(row[col]).strip()
                        })
                
                if text_data['texts']:
                    clinical_texts.append(text_data)
        
        logger.info(f"Extracted {len(clinical_texts)} clinical text records")
        return clinical_texts
    
    def get_icd_descriptions(self) -> Dict[str, str]:
        """Load ICD code descriptions"""
        icd_descriptions = {}
        
        # Load ICD diagnoses descriptions
        diag_df = self.load_csv('d_icd_diagnoses')
        if not diag_df.empty:
            for _, row in diag_df.iterrows():
                icd_descriptions[row['icd_code']] = row['long_title']
        
        # Load ICD procedures descriptions
        proc_df = self.load_csv('d_icd_procedures')
        if not proc_df.empty:
            for _, row in proc_df.iterrows():
                icd_descriptions[row['icd_code']] = row['long_title']
        
        return icd_descriptions
    
    def get_sample_records(self, n_patients: int = 10) -> List[Dict]:
        """Get sample patient records for testing"""
        patients_df = self.load_csv('patients', nrows=n_patients)
        sample_records = []
        
        for _, patient in patients_df.iterrows():
            subject_id = patient['subject_id']
            patient_data = self.get_patient_data(subject_id)
            sample_records.append(patient_data)
        
        return sample_records