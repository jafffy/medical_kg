import networkx as nx
from typing import Dict, List, Optional, Tuple, Set
import logging
from soap_kg.models.soap_schema import (
    SOAPKnowledgeGraph, MedicalEntity, MedicalRelation, 
    SOAPNote, EntityType, RelationType, SOAPCategory
)
import json
import pickle

logger = logging.getLogger(__name__)

class KnowledgeGraphBuilder:
    def __init__(self):
        self.soap_kg = SOAPKnowledgeGraph()
        self.networkx_graph = nx.MultiDiGraph()
        
    def add_entities(self, entities: List[MedicalEntity]):
        """Add entities to the knowledge graph"""
        for entity in entities:
            self.soap_kg.add_entity(entity)
            
            # Add to NetworkX graph
            self.networkx_graph.add_node(
                entity.id,
                text=entity.text,
                entity_type=entity.entity_type.value,
                soap_category=entity.soap_category.value,
                confidence=entity.confidence,
                **entity.metadata
            )
        
        logger.info(f"Added {len(entities)} entities to knowledge graph")
    
    def add_relationships(self, relationships: List[MedicalRelation]):
        """Add relationships to the knowledge graph"""
        for relation in relationships:
            self.soap_kg.add_relation(relation)
            
            # Add to NetworkX graph
            if (relation.source_entity in self.networkx_graph and 
                relation.target_entity in self.networkx_graph):
                
                self.networkx_graph.add_edge(
                    relation.source_entity,
                    relation.target_entity,
                    key=relation.id,
                    relation_type=relation.relation_type.value,
                    confidence=relation.confidence,
                    soap_context=relation.soap_context.value,
                    **relation.metadata
                )
        
        logger.info(f"Added {len(relationships)} relationships to knowledge graph")
    
    def add_soap_note(self, soap_note: SOAPNote):
        """Add a complete SOAP note to the knowledge graph"""
        self.soap_kg.add_soap_note(soap_note)
        
        # Add all entities and relationships from the SOAP note
        all_entities = soap_note.get_all_entities()
        self.add_entities(all_entities)
        self.add_relationships(soap_note.relations)
        
        logger.info(f"Added SOAP note for patient {soap_note.patient_id}")
    
    def get_subgraph_by_patient(self, patient_id: str) -> nx.MultiDiGraph:
        """Extract subgraph for a specific patient"""
        patient_entities = self.soap_kg.get_patient_entities(patient_id)
        entity_ids = [e.id for e in patient_entities]
        
        return self.networkx_graph.subgraph(entity_ids).copy()
    
    def get_subgraph_by_soap_category(self, category: SOAPCategory) -> nx.MultiDiGraph:
        """Extract subgraph for a specific SOAP category"""
        category_entities = self.soap_kg.get_entities_by_soap_category(category)
        entity_ids = [e.id for e in category_entities]
        
        return self.networkx_graph.subgraph(entity_ids).copy()
    
    def get_subgraph_by_entity_type(self, entity_type: EntityType) -> nx.MultiDiGraph:
        """Extract subgraph for a specific entity type"""
        matching_entities = [
            e for e in self.soap_kg.entities.values() 
            if e.entity_type == entity_type
        ]
        entity_ids = [e.id for e in matching_entities]
        
        return self.networkx_graph.subgraph(entity_ids).copy()
    
    def find_shortest_path(self, source_entity_id: str, target_entity_id: str) -> List[str]:
        """Find shortest path between two entities"""
        try:
            # Convert to undirected graph for path finding
            undirected = self.networkx_graph.to_undirected()
            path = nx.shortest_path(undirected, source_entity_id, target_entity_id)
            return path
        except nx.NetworkXNoPath:
            logger.warning(f"No path found between {source_entity_id} and {target_entity_id}")
            return []
        except Exception as e:
            logger.error(f"Error finding path: {e}")
            return []
    
    def get_entity_neighbors(self, entity_id: str, max_distance: int = 1) -> Dict:
        """Get neighboring entities within specified distance"""
        if entity_id not in self.networkx_graph:
            return {}
        
        neighbors = {}
        for distance in range(1, max_distance + 1):
            if distance == 1:
                # Direct neighbors
                direct_neighbors = list(self.networkx_graph.neighbors(entity_id))
                neighbors[distance] = direct_neighbors
            else:
                # Neighbors at specific distance
                ego_graph = nx.ego_graph(self.networkx_graph, entity_id, radius=distance)
                distance_nodes = []
                for node in ego_graph.nodes():
                    try:
                        if nx.shortest_path_length(ego_graph, entity_id, node) == distance:
                            distance_nodes.append(node)
                    except nx.NetworkXNoPath:
                        continue
                neighbors[distance] = distance_nodes
        
        return neighbors
    
    def calculate_centrality_metrics(self) -> Dict:
        """Calculate various centrality metrics for entities"""
        if self.networkx_graph.number_of_nodes() == 0:
            return {}
        
        # Convert to undirected for some metrics
        undirected = self.networkx_graph.to_undirected()
        
        metrics = {}
        
        try:
            # Degree centrality
            metrics['degree_centrality'] = nx.degree_centrality(undirected)
            
            # Betweenness centrality (sample if graph is large)
            if undirected.number_of_nodes() > 1000:
                k = min(1000, undirected.number_of_nodes())
                metrics['betweenness_centrality'] = nx.betweenness_centrality(undirected, k=k)
            else:
                metrics['betweenness_centrality'] = nx.betweenness_centrality(undirected)
            
            # Closeness centrality
            metrics['closeness_centrality'] = nx.closeness_centrality(undirected)
            
            # PageRank
            metrics['pagerank'] = nx.pagerank(self.networkx_graph)
            
        except Exception as e:
            logger.error(f"Error calculating centrality metrics: {e}")
        
        return metrics
    
    def detect_communities(self) -> List[Set[str]]:
        """Detect communities in the knowledge graph"""
        try:
            undirected = self.networkx_graph.to_undirected()
            communities = nx.community.greedy_modularity_communities(undirected)
            return [set(community) for community in communities]
        except Exception as e:
            logger.error(f"Error detecting communities: {e}")
            return []
    
    def get_graph_statistics(self) -> Dict:
        """Get comprehensive statistics about the knowledge graph"""
        stats = self.soap_kg.get_statistics()
        
        # Add NetworkX graph statistics
        stats.update({
            'networkx_nodes': self.networkx_graph.number_of_nodes(),
            'networkx_edges': self.networkx_graph.number_of_edges(),
            'is_connected': nx.is_connected(self.networkx_graph.to_undirected()),
            'number_of_components': nx.number_connected_components(self.networkx_graph.to_undirected()),
            'density': nx.density(self.networkx_graph),
            'average_clustering': self._safe_average_clustering()
        })
        
        # Add degree distribution
        degrees = [d for n, d in self.networkx_graph.degree()]
        if degrees:
            stats['degree_distribution'] = {
                'min': min(degrees),
                'max': max(degrees),
                'mean': sum(degrees) / len(degrees),
                'median': sorted(degrees)[len(degrees) // 2]
            }
        
        return stats
    
    def _safe_average_clustering(self) -> float:
        """Safely calculate average clustering coefficient"""
        try:
            if self.networkx_graph.number_of_nodes() == 0:
                return 0.0
            
            # Convert to simple undirected graph for clustering calculation
            simple_graph = nx.Graph()
            for node in self.networkx_graph.nodes():
                simple_graph.add_node(node)
            
            for edge in self.networkx_graph.edges():
                simple_graph.add_edge(edge[0], edge[1])
            
            return nx.average_clustering(simple_graph)
        except Exception as e:
            logger.error(f"Error calculating clustering coefficient: {e}")
            return 0.0
    
    def export_to_formats(self, base_filename: str):
        """Export knowledge graph to various formats"""
        try:
            # Export SOAP KG to JSON
            def make_serializable(obj):
                """Convert objects to JSON serializable format"""
                if hasattr(obj, 'value'):  # Enum types
                    return obj.value
                elif isinstance(obj, dict):
                    return {k: make_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [make_serializable(item) for item in obj]
                elif isinstance(obj, set):
                    return list(obj)
                else:
                    return obj
            
            kg_data = {
                'entities': {
                    eid: {
                        'text': e.text,
                        'entity_type': e.entity_type.value,
                        'soap_category': e.soap_category.value,
                        'confidence': e.confidence,
                        'metadata': make_serializable(e.metadata)
                    }
                    for eid, e in self.soap_kg.entities.items()
                },
                'relations': {
                    rid: {
                        'source_entity': r.source_entity,
                        'target_entity': r.target_entity,
                        'relation_type': r.relation_type.value,
                        'confidence': r.confidence,
                        'soap_context': r.soap_context.value,
                        'metadata': make_serializable(r.metadata)
                    }
                    for rid, r in self.soap_kg.relations.items()
                },
                'statistics': make_serializable(self.get_graph_statistics())
            }
            
            with open(f"{base_filename}_soap_kg.json", 'w') as f:
                json.dump(kg_data, f, indent=2)
            
            # Export NetworkX graph
            nx.write_gexf(self.networkx_graph, f"{base_filename}_networkx.gexf")
            nx.write_graphml(self.networkx_graph, f"{base_filename}_networkx.graphml")
            
            # Export as pickle for Python use
            with open(f"{base_filename}_soap_kg.pkl", 'wb') as f:
                pickle.dump(self.soap_kg, f)
            
            logger.info(f"Exported knowledge graph to multiple formats with base name: {base_filename}")
            
        except Exception as e:
            logger.error(f"Error exporting knowledge graph: {e}")
    
    def load_from_file(self, filename: str):
        """Load knowledge graph from pickle file"""
        try:
            with open(filename, 'rb') as f:
                self.soap_kg = pickle.load(f)
            
            # Rebuild NetworkX graph
            self.networkx_graph = nx.MultiDiGraph()
            
            # Add entities
            for entity in self.soap_kg.entities.values():
                self.networkx_graph.add_node(
                    entity.id,
                    text=entity.text,
                    entity_type=entity.entity_type.value,
                    soap_category=entity.soap_category.value,
                    confidence=entity.confidence,
                    **entity.metadata
                )
            
            # Add relationships
            for relation in self.soap_kg.relations.values():
                if (relation.source_entity in self.networkx_graph and 
                    relation.target_entity in self.networkx_graph):
                    
                    self.networkx_graph.add_edge(
                        relation.source_entity,
                        relation.target_entity,
                        key=relation.id,
                        relation_type=relation.relation_type.value,
                        confidence=relation.confidence,
                        soap_context=relation.soap_context.value,
                        **relation.metadata
                    )
            
            logger.info(f"Loaded knowledge graph from {filename}")
            
        except Exception as e:
            logger.error(f"Error loading knowledge graph: {e}")
    
    def query_entities(self, query_text: str, entity_type: EntityType = None, 
                      soap_category: SOAPCategory = None, 
                      min_confidence: float = 0.0) -> List[MedicalEntity]:
        """Query entities by text, type, category, and confidence"""
        results = []
        
        for entity in self.soap_kg.entities.values():
            # Text matching
            if query_text.lower() not in entity.text.lower():
                continue
            
            # Type filtering
            if entity_type and entity.entity_type != entity_type:
                continue
            
            # SOAP category filtering
            if soap_category and entity.soap_category != soap_category:
                continue
            
            # Confidence filtering
            if entity.confidence < min_confidence:
                continue
            
            results.append(entity)
        
        return results
    
    def get_related_entities(self, entity_id: str, relation_types: List[RelationType] = None) -> List[Tuple[MedicalEntity, MedicalRelation]]:
        """Get entities related to a given entity through specific relation types"""
        related = []
        
        for relation in self.soap_kg.relations.values():
            if relation_types and relation.relation_type not in relation_types:
                continue
            
            target_entity_id = None
            if relation.source_entity == entity_id:
                target_entity_id = relation.target_entity
            elif relation.target_entity == entity_id:
                target_entity_id = relation.source_entity
            
            if target_entity_id and target_entity_id in self.soap_kg.entities:
                target_entity = self.soap_kg.entities[target_entity_id]
                related.append((target_entity, relation))
        
        return related