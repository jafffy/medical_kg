from dataclasses import dataclass
from typing import List, Dict, Optional, Set
from enum import Enum

class SOAPCategory(Enum):
    SUBJECTIVE = "subjective"
    OBJECTIVE = "objective"
    ASSESSMENT = "assessment"
    PLAN = "plan"

class EntityType(Enum):
    DISEASE = "disease"
    SYMPTOM = "symptom"
    MEDICATION = "medication"
    PROCEDURE = "procedure"
    ANATOMY = "anatomy"
    LAB_VALUE = "lab_value"
    VITAL_SIGN = "vital_sign"
    TREATMENT = "treatment"
    DEMOGRAPHIC = "demographic"

class RelationType(Enum):
    TREATS = "treats"
    CAUSES = "causes"
    INDICATES = "indicates"
    MEASURED_BY = "measured_by"
    LOCATED_IN = "located_in"
    HAS_SYMPTOM = "has_symptom"
    PRESCRIBED_FOR = "prescribed_for"
    DIAGNOSED_WITH = "diagnosed_with"
    PART_OF = "part_of"
    FOLLOWS = "follows"

@dataclass
class MedicalEntity:
    id: str
    text: str
    entity_type: EntityType
    soap_category: SOAPCategory
    confidence: float
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class MedicalRelation:
    id: str
    source_entity: str
    target_entity: str
    relation_type: RelationType
    confidence: float
    soap_context: SOAPCategory
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class SOAPNote:
    patient_id: str
    admission_id: Optional[str]
    subjective: List[MedicalEntity]
    objective: List[MedicalEntity]
    assessment: List[MedicalEntity]
    plan: List[MedicalEntity]
    relations: List[MedicalRelation]
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def get_all_entities(self) -> List[MedicalEntity]:
        return self.subjective + self.objective + self.assessment + self.plan
    
    def get_entities_by_type(self, entity_type: EntityType) -> List[MedicalEntity]:
        return [e for e in self.get_all_entities() if e.entity_type == entity_type]

class SOAPKnowledgeGraph:
    def __init__(self):
        self.entities: Dict[str, MedicalEntity] = {}
        self.relations: Dict[str, MedicalRelation] = {}
        self.soap_notes: Dict[str, SOAPNote] = {}
        self.entity_types: Set[EntityType] = set()
        self.relation_types: Set[RelationType] = set()
    
    def add_entity(self, entity: MedicalEntity):
        self.entities[entity.id] = entity
        self.entity_types.add(entity.entity_type)
    
    def add_relation(self, relation: MedicalRelation):
        self.relations[relation.id] = relation
        self.relation_types.add(relation.relation_type)
    
    def add_soap_note(self, soap_note: SOAPNote):
        self.soap_notes[soap_note.patient_id] = soap_note
        
        for entity in soap_note.get_all_entities():
            self.add_entity(entity)
        
        for relation in soap_note.relations:
            self.add_relation(relation)
    
    def get_patient_entities(self, patient_id: str) -> List[MedicalEntity]:
        if patient_id not in self.soap_notes:
            return []
        return self.soap_notes[patient_id].get_all_entities()
    
    def get_entities_by_soap_category(self, category: SOAPCategory) -> List[MedicalEntity]:
        return [e for e in self.entities.values() if e.soap_category == category]
    
    def get_relations_by_type(self, relation_type: RelationType) -> List[MedicalRelation]:
        return [r for r in self.relations.values() if r.relation_type == relation_type]
    
    def get_entity_neighbors(self, entity_id: str) -> List[str]:
        neighbors = []
        for relation in self.relations.values():
            if relation.source_entity == entity_id:
                neighbors.append(relation.target_entity)
            elif relation.target_entity == entity_id:
                neighbors.append(relation.source_entity)
        return neighbors
    
    def get_statistics(self) -> Dict:
        return {
            "total_entities": len(self.entities),
            "total_relations": len(self.relations),
            "total_patients": len(self.soap_notes),
            "entity_types": list(self.entity_types),
            "relation_types": list(self.relation_types),
            "soap_distribution": {
                category.value: len(self.get_entities_by_soap_category(category))
                for category in SOAPCategory
            }
        }