import re
import string
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class MedicalTextPreprocessor:
    def __init__(self):
        self.medical_abbreviations = {
            'bp': 'blood pressure',
            'hr': 'heart rate',
            'rr': 'respiratory rate',
            'temp': 'temperature',
            'o2': 'oxygen',
            'co2': 'carbon dioxide',
            'hx': 'history',
            'dx': 'diagnosis',
            'tx': 'treatment',
            'rx': 'prescription',
            'pt': 'patient',
            'pts': 'patients',
            'yo': 'year old',
            'yom': 'year old male',
            'yof': 'year old female',
            'w/': 'with',
            'w/o': 'without',
            'c/o': 'complains of',
            's/p': 'status post',
            'h/o': 'history of',
            'r/o': 'rule out',
            'sob': 'shortness of breath',
            'cp': 'chest pain',
            'n/v': 'nausea and vomiting',
            'abd': 'abdominal',
            'ext': 'extremities',
            'neuro': 'neurological',
            'psych': 'psychiatric',
            'gi': 'gastrointestinal',
            'gu': 'genitourinary',
            'ent': 'ear nose throat',
            'cvs': 'cardiovascular system',
            'cns': 'central nervous system',
            'pns': 'peripheral nervous system'
        }
        
        self.units_pattern = re.compile(r'\b(\d+(?:\.\d+)?)\s*(mg|ml|mcg|g|kg|l|dl|mmol|mmhg|bpm|rpm|celsius|fahrenheit|f|c)\b', re.IGNORECASE)
        self.time_pattern = re.compile(r'\b(\d{1,2}):(\d{2})\s*(am|pm)?\b', re.IGNORECASE)
        self.date_pattern = re.compile(r'\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b')
        
    def expand_abbreviations(self, text: str) -> str:
        """Expand common medical abbreviations"""
        text_lower = text.lower()
        for abbrev, expansion in self.medical_abbreviations.items():
            pattern = r'\b' + re.escape(abbrev) + r'\b'
            text_lower = re.sub(pattern, expansion, text_lower)
        return text_lower
    
    def normalize_units(self, text: str) -> str:
        """Normalize medical units and measurements"""
        def unit_replacer(match):
            value, unit = match.groups()
            unit_normalized = unit.lower()
            
            # Normalize common unit variations
            unit_map = {
                'mg': 'milligrams',
                'ml': 'milliliters', 
                'mcg': 'micrograms',
                'g': 'grams',
                'kg': 'kilograms',
                'l': 'liters',
                'dl': 'deciliters',
                'mmol': 'millimoles',
                'mmhg': 'mmHg',
                'bpm': 'beats per minute',
                'rpm': 'respirations per minute',
                'celsius': 'degrees Celsius',
                'fahrenheit': 'degrees Fahrenheit',
                'f': 'degrees Fahrenheit',
                'c': 'degrees Celsius'
            }
            
            normalized_unit = unit_map.get(unit_normalized, unit_normalized)
            return f"{value} {normalized_unit}"
        
        return self.units_pattern.sub(unit_replacer, text)
    
    def clean_text(self, text: str) -> str:
        """Clean and standardize medical text"""
        if not text or not isinstance(text, str):
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove special characters but keep medical-relevant punctuation
        text = re.sub(r'[^\w\s\-\./,;:()\[\]%]', '', text)
        
        # Normalize case - keep as is for medical terms
        # text = text.lower()
        
        # Remove very short tokens (likely noise)
        words = text.split()
        words = [word for word in words if len(word) > 1 or word.isdigit()]
        
        return ' '.join(words)
    
    def extract_vital_signs(self, text: str) -> List[Dict]:
        """Extract vital signs patterns from text"""
        vital_patterns = {
            'blood_pressure': r'(?:bp|blood pressure)\s*:?\s*(\d{2,3})/(\d{2,3})',
            'heart_rate': r'(?:hr|heart rate|pulse)\s*:?\s*(\d{2,3})\s*(?:bpm)?',
            'respiratory_rate': r'(?:rr|resp rate|respiratory rate)\s*:?\s*(\d{1,2})',
            'temperature': r'(?:temp|temperature)\s*:?\s*(\d{2,3}(?:\.\d)?)\s*(?:f|c|celsius|fahrenheit)?',
            'oxygen_saturation': r'(?:o2 sat|oxygen saturation|spo2)\s*:?\s*(\d{2,3})%?',
            'weight': r'(?:weight|wt)\s*:?\s*(\d{2,3}(?:\.\d)?)\s*(?:kg|lbs|pounds)?',
            'height': r'(?:height|ht)\s*:?\s*(\d{1,3})\s*(?:cm|inches|in|ft)?'
        }
        
        vitals = []
        for vital_type, pattern in vital_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if vital_type == 'blood_pressure':
                    systolic, diastolic = match.groups()
                    vitals.append({
                        'type': 'systolic_bp',
                        'value': systolic,
                        'text': match.group(0)
                    })
                    vitals.append({
                        'type': 'diastolic_bp', 
                        'value': diastolic,
                        'text': match.group(0)
                    })
                else:
                    vitals.append({
                        'type': vital_type,
                        'value': match.group(1),
                        'text': match.group(0)
                    })
        
        return vitals
    
    def extract_medications(self, text: str) -> List[Dict]:
        """Extract medication patterns from text"""
        # Common medication patterns
        med_patterns = [
            r'\b(\w+)\s+(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|units?)\b',  # drug dose unit
            r'\b(\w+)\s+(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|units?)\s+(?:po|iv|im|sq|pr)\b',  # with route
            r'\b(\w+)\s+tablet\b',  # tablet form
            r'\b(\w+)\s+capsule\b',  # capsule form
        ]
        
        medications = []
        for pattern in med_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                medications.append({
                    'medication': match.group(1),
                    'full_text': match.group(0),
                    'dose': match.group(2) if len(match.groups()) > 1 else None,
                    'unit': match.group(3) if len(match.groups()) > 2 else None
                })
        
        return medications
    
    def segment_sentences(self, text: str) -> List[str]:
        """Segment text into sentences for processing"""
        # Handle medical text sentence boundaries
        text = re.sub(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', '|SENT|', text)
        sentences = [s.strip() for s in text.split('|SENT|') if s.strip()]
        return sentences
    
    def preprocess_clinical_text(self, text: str) -> Dict:
        """Complete preprocessing pipeline for clinical text"""
        if not text:
            return {
                'original_text': text,
                'cleaned_text': '',
                'sentences': [],
                'vital_signs': [],
                'medications': [],
                'metadata': {}
            }
        
        # Clean and normalize
        cleaned = self.clean_text(text)
        expanded = self.expand_abbreviations(cleaned)
        normalized = self.normalize_units(expanded)
        
        # Extract structured information
        sentences = self.segment_sentences(normalized)
        vital_signs = self.extract_vital_signs(text)  # Use original for better pattern matching
        medications = self.extract_medications(text)  # Use original for better pattern matching
        
        return {
            'original_text': text,
            'cleaned_text': normalized,
            'sentences': sentences,
            'vital_signs': vital_signs,
            'medications': medications,
            'metadata': {
                'sentence_count': len(sentences),
                'vital_signs_count': len(vital_signs),
                'medications_count': len(medications),
                'character_count': len(normalized),
                'word_count': len(normalized.split())
            }
        }