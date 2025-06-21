import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from knowledge_graph_builder import KnowledgeGraphBuilder
from soap_schema import SOAPCategory, EntityType, RelationType
import numpy as np

logger = logging.getLogger(__name__)

class KnowledgeGraphVisualizer:
    def __init__(self, kg_builder: KnowledgeGraphBuilder):
        self.kg_builder = kg_builder
        self.soap_colors = {
            SOAPCategory.SUBJECTIVE: '#FF6B6B',    # Red
            SOAPCategory.OBJECTIVE: '#4ECDC4',     # Teal
            SOAPCategory.ASSESSMENT: '#45B7D1',    # Blue
            SOAPCategory.PLAN: '#96CEB4'           # Green
        }
        
        self.entity_colors = {
            EntityType.DISEASE: '#E74C3C',
            EntityType.SYMPTOM: '#F39C12',
            EntityType.MEDICATION: '#3498DB',
            EntityType.PROCEDURE: '#9B59B6',
            EntityType.ANATOMY: '#2ECC71',
            EntityType.LAB_VALUE: '#E67E22',
            EntityType.VITAL_SIGN: '#1ABC9C',
            EntityType.TREATMENT: '#34495E'
        }
    
    def plot_soap_distribution(self, save_path: str = None) -> None:
        """Plot distribution of entities across SOAP categories"""
        stats = self.kg_builder.get_graph_statistics()
        soap_dist = stats.get('soap_distribution', {})
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar chart
        categories = list(soap_dist.keys())
        counts = list(soap_dist.values())
        colors = [self.soap_colors.get(SOAPCategory(cat), '#CCCCCC') for cat in categories]
        
        ax1.bar(categories, counts, color=colors)
        ax1.set_title('SOAP Category Distribution')
        ax1.set_xlabel('SOAP Category')
        ax1.set_ylabel('Number of Entities')
        ax1.tick_params(axis='x', rotation=45)
        
        # Pie chart
        ax2.pie(counts, labels=categories, colors=colors, autopct='%1.1f%%')
        ax2.set_title('SOAP Category Proportions')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"SOAP distribution plot saved to {save_path}")
        else:
            plt.show()
    
    def plot_entity_type_distribution(self, save_path: str = None) -> None:
        """Plot distribution of entity types"""
        entity_type_counts = {}
        
        for entity in self.kg_builder.soap_kg.entities.values():
            entity_type = entity.entity_type
            entity_type_counts[entity_type] = entity_type_counts.get(entity_type, 0) + 1
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        types = list(entity_type_counts.keys())
        counts = list(entity_type_counts.values())
        colors = [self.entity_colors.get(t, '#CCCCCC') for t in types]
        
        ax.barh([t.value for t in types], counts, color=colors)
        ax.set_title('Entity Type Distribution')
        ax.set_xlabel('Number of Entities')
        ax.set_ylabel('Entity Type')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Entity type distribution plot saved to {save_path}")
        else:
            plt.show()
    
    def plot_network_graph(self, layout: str = 'spring', max_nodes: int = 100, 
                          save_path: str = None) -> None:
        """Plot the knowledge graph network"""
        graph = self.kg_builder.networkx_graph
        
        if graph.number_of_nodes() > max_nodes:
            # Sample nodes if graph is too large
            nodes_to_show = list(graph.nodes())[:max_nodes]
            graph = graph.subgraph(nodes_to_show)
            logger.info(f"Showing subgraph with {len(nodes_to_show)} nodes")
        
        fig, ax = plt.subplots(figsize=(15, 15))
        
        # Choose layout
        if layout == 'spring':
            pos = nx.spring_layout(graph, k=1, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(graph)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(graph)
        else:
            pos = nx.spring_layout(graph)
        
        # Get node colors based on SOAP categories
        node_colors = []
        for node in graph.nodes():
            soap_cat = graph.nodes[node].get('soap_category', 'objective')
            try:
                category = SOAPCategory(soap_cat)
                color = self.soap_colors.get(category, '#CCCCCC')
            except ValueError:
                color = '#CCCCCC'
            node_colors.append(color)
        
        # Draw network
        nx.draw_networkx_nodes(graph, pos, node_color=node_colors, 
                              node_size=300, alpha=0.8, ax=ax)
        nx.draw_networkx_edges(graph, pos, alpha=0.3, edge_color='gray', 
                              width=0.5, ax=ax)
        
        # Add labels for important nodes (high degree)
        degrees = dict(graph.degree())
        high_degree_nodes = [node for node, degree in degrees.items() 
                           if degree > np.percentile(list(degrees.values()), 90)]
        
        labels = {node: graph.nodes[node].get('text', '')[:15] + '...' 
                 for node in high_degree_nodes}
        nx.draw_networkx_labels(graph, pos, labels, font_size=8, ax=ax)
        
        ax.set_title('SOAP Knowledge Graph Network')
        ax.axis('off')
        
        # Add legend
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor=color, markersize=10, label=cat.value)
                         for cat, color in self.soap_colors.items()]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Network graph saved to {save_path}")
        else:
            plt.show()
    
    def create_interactive_network(self, max_nodes: int = 200, save_path: str = None):
        """Create interactive network visualization using Plotly"""
        graph = self.kg_builder.networkx_graph
        
        if graph.number_of_nodes() > max_nodes:
            # Sample nodes if graph is too large
            nodes_to_show = list(graph.nodes())[:max_nodes]
            graph = graph.subgraph(nodes_to_show)
        
        # Create layout
        pos = nx.spring_layout(graph, k=1, iterations=50)
        
        # Extract node information
        node_x = [pos[node][0] for node in graph.nodes()]
        node_y = [pos[node][1] for node in graph.nodes()]
        
        node_info = []
        node_colors = []
        
        for node in graph.nodes():
            node_data = graph.nodes[node]
            text = node_data.get('text', 'Unknown')
            entity_type = node_data.get('entity_type', 'unknown')
            soap_category = node_data.get('soap_category', 'objective')
            confidence = node_data.get('confidence', 0.0)
            
            node_info.append(f"Text: {text}<br>"
                           f"Type: {entity_type}<br>"
                           f"SOAP: {soap_category}<br>"
                           f"Confidence: {confidence:.2f}")
            
            try:
                category = SOAPCategory(soap_category)
                color = self.soap_colors.get(category, '#CCCCCC')
            except ValueError:
                color = '#CCCCCC'
            node_colors.append(color)
        
        # Extract edge information
        edge_x = []
        edge_y = []
        
        for edge in graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        # Create traces
        edge_trace = go.Scatter(x=edge_x, y=edge_y,
                               line=dict(width=0.5, color='#888'),
                               hoverinfo='none',
                               mode='lines')
        
        node_trace = go.Scatter(x=node_x, y=node_y,
                               mode='markers',
                               hoverinfo='text',
                               text=node_info,
                               marker=dict(size=10,
                                         color=node_colors,
                                         line=dict(width=2, color='white')))
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title='Interactive SOAP Knowledge Graph',
                           titlefont_size=16,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[ dict(
                               text="Hover over nodes for details",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002,
                               xanchor='left', yanchor='bottom',
                               font=dict(color='#888', size=12)
                           )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Interactive network saved to {save_path}")
        else:
            fig.show()
    
    def plot_relationship_matrix(self, save_path: str = None) -> None:
        """Plot relationship type matrix"""
        relations = self.kg_builder.soap_kg.relations.values()
        
        # Count relationships by type
        rel_counts = {}
        for relation in relations:
            rel_type = relation.relation_type
            rel_counts[rel_type] = rel_counts.get(rel_type, 0) + 1
        
        # Create matrix plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        types = list(rel_counts.keys())
        counts = list(rel_counts.values())
        
        ax.bar([t.value for t in types], counts, color='skyblue')
        ax.set_title('Relationship Type Distribution')
        ax.set_xlabel('Relationship Type')
        ax.set_ylabel('Number of Relations')
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Relationship matrix saved to {save_path}")
        else:
            plt.show()
    
    def plot_confidence_distribution(self, save_path: str = None) -> None:
        """Plot confidence score distributions"""
        entity_confidences = [e.confidence for e in self.kg_builder.soap_kg.entities.values()]
        relation_confidences = [r.confidence for r in self.kg_builder.soap_kg.relations.values()]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Entity confidence distribution
        ax1.hist(entity_confidences, bins=20, alpha=0.7, color='blue', label='Entities')
        ax1.set_title('Entity Confidence Distribution')
        ax1.set_xlabel('Confidence Score')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        
        # Relationship confidence distribution
        ax2.hist(relation_confidences, bins=20, alpha=0.7, color='red', label='Relations')
        ax2.set_title('Relationship Confidence Distribution')
        ax2.set_xlabel('Confidence Score')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confidence distribution plot saved to {save_path}")
        else:
            plt.show()
    
    def create_dashboard(self, save_path: str = None):
        """Create comprehensive dashboard with multiple visualizations"""
        stats = self.kg_builder.get_graph_statistics()
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('SOAP Distribution', 'Entity Types', 
                          'Relationship Types', 'Graph Statistics'),
            specs=[[{"type": "pie"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "table"}]]
        )
        
        # SOAP distribution pie chart
        soap_dist = stats.get('soap_distribution', {})
        fig.add_trace(
            go.Pie(labels=list(soap_dist.keys()), 
                   values=list(soap_dist.values()),
                   name="SOAP"),
            row=1, col=1
        )
        
        # Entity type distribution
        entity_type_counts = {}
        for entity in self.kg_builder.soap_kg.entities.values():
            entity_type = entity.entity_type.value
            entity_type_counts[entity_type] = entity_type_counts.get(entity_type, 0) + 1
        
        fig.add_trace(
            go.Bar(x=list(entity_type_counts.keys()), 
                   y=list(entity_type_counts.values()),
                   name="Entity Types"),
            row=1, col=2
        )
        
        # Relationship type distribution
        rel_counts = {}
        for relation in self.kg_builder.soap_kg.relations.values():
            rel_type = relation.relation_type.value
            rel_counts[rel_type] = rel_counts.get(rel_type, 0) + 1
        
        fig.add_trace(
            go.Bar(x=list(rel_counts.keys()), 
                   y=list(rel_counts.values()),
                   name="Relation Types"),
            row=2, col=1
        )
        
        # Statistics table
        table_data = [
            ['Total Entities', stats.get('total_entities', 0)],
            ['Total Relations', stats.get('total_relations', 0)],
            ['Total Patients', stats.get('total_patients', 0)],
            ['Graph Density', f"{stats.get('density', 0):.3f}"],
            ['Avg Clustering', f"{stats.get('average_clustering', 0):.3f}"],
            ['Connected Components', stats.get('number_of_components', 0)]
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Metric', 'Value']),
                cells=dict(values=list(zip(*table_data)))
            ),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title_text="SOAP Knowledge Graph Dashboard")
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Dashboard saved to {save_path}")
        else:
            fig.show()
    
    def visualize_patient_journey(self, patient_id: str, save_path: str = None):
        """Visualize a specific patient's medical journey through the knowledge graph"""
        patient_subgraph = self.kg_builder.get_subgraph_by_patient(patient_id)
        
        if patient_subgraph.number_of_nodes() == 0:
            logger.warning(f"No data found for patient {patient_id}")
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create layout
        pos = nx.spring_layout(patient_subgraph, k=2, iterations=50)
        
        # Draw nodes colored by SOAP category
        node_colors = []
        for node in patient_subgraph.nodes():
            soap_cat = patient_subgraph.nodes[node].get('soap_category', 'objective')
            try:
                category = SOAPCategory(soap_cat)
                color = self.soap_colors.get(category, '#CCCCCC')
            except ValueError:
                color = '#CCCCCC'
            node_colors.append(color)
        
        nx.draw_networkx_nodes(patient_subgraph, pos, node_color=node_colors, 
                              node_size=500, alpha=0.8, ax=ax)
        nx.draw_networkx_edges(patient_subgraph, pos, alpha=0.5, 
                              edge_color='gray', width=1, ax=ax)
        
        # Add labels
        labels = {node: patient_subgraph.nodes[node].get('text', '')[:20] + '...' 
                 for node in patient_subgraph.nodes()}
        nx.draw_networkx_labels(patient_subgraph, pos, labels, font_size=8, ax=ax)
        
        ax.set_title(f'Patient {patient_id} Medical Journey')
        ax.axis('off')
        
        # Add legend
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor=color, markersize=10, label=cat.value)
                         for cat, color in self.soap_colors.items()]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Patient journey visualization saved to {save_path}")
        else:
            plt.show()