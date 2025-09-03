"""
Knowledge extraction utilities for engineering applications.
"""

import logging
from typing import Dict, Any, List, Optional, Union, Tuple
import re
import json
from collections import defaultdict, Counter
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class KnowledgeExtractor:
    """
    Basic knowledge extraction utilities.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.extracted_knowledge = defaultdict(list)
        self.knowledge_graph = nx.DiGraph()
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities from text."""
        entities = []
        
        # Pattern for measurements
        measurement_pattern = r'(\d+(?:\.\d+)?)\s*(mm|cm|m|in|ft|MPa|GPa|psi|ksi|N|kN|MN|kg|g|V|A|W|kW|MW|°C|°F|K|rpm|Hz|kHz|MHz|GHz)'
        measurements = re.findall(measurement_pattern, text, re.IGNORECASE)
        
        for value, unit in measurements:
            entities.append({
                "type": "measurement",
                "value": float(value),
                "unit": unit,
                "text": f"{value} {unit}"
            })
        
        # Pattern for standards
        standard_pattern = r'(ASTM|ISO|ANSI|BS|DIN|JIS)\s*([A-Z0-9\-]+)'
        standards = re.findall(standard_pattern, text, re.IGNORECASE)
        
        for org, code in standards:
            entities.append({
                "type": "standard",
                "organization": org.upper(),
                "code": code,
                "text": f"{org} {code}"
            })
        
        # Pattern for materials
        material_pattern = r'\b(steel|aluminum|titanium|concrete|composite|polymer|ceramic|alloy|plastic|rubber|glass|wood|carbon|fiber|resin|epoxy)\b'
        materials = re.findall(material_pattern, text, re.IGNORECASE)
        
        for material in materials:
            entities.append({
                "type": "material",
                "name": material.lower(),
                "text": material
            })
        
        return entities
    
    def extract_relationships(self, text: str) -> List[Dict[str, Any]]:
        """Extract relationships between entities."""
        relationships = []
        
        # Pattern for "X is Y" relationships
        is_pattern = r'([A-Za-z\s]+)\s+is\s+([A-Za-z\s]+)'
        is_matches = re.findall(is_pattern, text, re.IGNORECASE)
        
        for subject, object_ in is_matches:
            relationships.append({
                "subject": subject.strip(),
                "predicate": "is",
                "object": object_.strip(),
                "confidence": 0.8
            })
        
        # Pattern for "X has Y" relationships
        has_pattern = r'([A-Za-z\s]+)\s+has\s+([A-Za-z\s]+)'
        has_matches = re.findall(has_pattern, text, re.IGNORECASE)
        
        for subject, object_ in has_matches:
            relationships.append({
                "subject": subject.strip(),
                "predicate": "has",
                "object": object_.strip(),
                "confidence": 0.7
            })
        
        # Pattern for "X requires Y" relationships
        requires_pattern = r'([A-Za-z\s]+)\s+requires\s+([A-Za-z\s]+)'
        requires_matches = re.findall(requires_pattern, text, re.IGNORECASE)
        
        for subject, object_ in requires_matches:
            relationships.append({
                "subject": subject.strip(),
                "predicate": "requires",
                "object": object_.strip(),
                "confidence": 0.7
            })
        
        return relationships
    
    def extract_concepts(self, text: str) -> List[Dict[str, Any]]:
        """Extract key concepts from text."""
        concepts = []
        
        # Engineering concept patterns
        concept_patterns = {
            "design": r'\b(design|designing|designed)\b',
            "analysis": r'\b(analysis|analyze|analyzing|analyzed)\b',
            "testing": r'\b(testing|test|tested)\b',
            "optimization": r'\b(optimization|optimize|optimizing|optimized)\b',
            "manufacturing": r'\b(manufacturing|manufacture|manufactured)\b',
            "quality": r'\b(quality|qualitative|quantitative)\b',
            "safety": r'\b(safety|safe|unsafe)\b',
            "performance": r'\b(performance|perform|performing|performed)\b',
            "reliability": r'\b(reliability|reliable|unreliable)\b',
            "efficiency": r'\b(efficiency|efficient|inefficient)\b'
        }
        
        for concept, pattern in concept_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                concepts.append({
                    "concept": concept,
                    "mentions": len(matches),
                    "confidence": min(len(matches) / 5.0, 1.0)
                })
        
        return concepts
    
    def build_knowledge_graph(self, entities: List[Dict[str, Any]], relationships: List[Dict[str, Any]]):
        """Build a knowledge graph from extracted entities and relationships."""
        # Add entities as nodes
        for entity in entities:
            node_id = f"{entity['type']}_{entity.get('text', entity.get('name', ''))}"
            self.knowledge_graph.add_node(node_id, **entity)
        
        # Add relationships as edges
        for rel in relationships:
            subject_id = self._find_entity_id(rel['subject'], entities)
            object_id = self._find_entity_id(rel['object'], entities)
            
            if subject_id and object_id:
                self.knowledge_graph.add_edge(
                    subject_id, 
                    object_id, 
                    predicate=rel['predicate'],
                    confidence=rel['confidence']
                )
    
    def _find_entity_id(self, text: str, entities: List[Dict[str, Any]]) -> Optional[str]:
        """Find entity ID by text match."""
        for entity in entities:
            if entity.get('text', '').lower() == text.lower():
                return f"{entity['type']}_{entity.get('text', entity.get('name', ''))}"
        return None
    
    def get_knowledge_summary(self) -> Dict[str, Any]:
        """Get summary of extracted knowledge."""
        return {
            "total_entities": len(self.knowledge_graph.nodes()),
            "total_relationships": len(self.knowledge_graph.edges()),
            "entity_types": list(set(node['type'] for node in self.knowledge_graph.nodes.values())),
            "relationship_types": list(set(edge['predicate'] for edge in self.knowledge_graph.edges.values())),
            "graph_density": nx.density(self.knowledge_graph)
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get extractor status."""
        return {
            "knowledge_graph_nodes": len(self.knowledge_graph.nodes()),
            "knowledge_graph_edges": len(self.knowledge_graph.edges()),
            "extracted_knowledge_types": list(self.extracted_knowledge.keys())
        }


class EngineeringKnowledgeExtractor(KnowledgeExtractor):
    """
    Specialized knowledge extractor for engineering applications.
    """
    
    def __init__(self):
        super().__init__()
        self.engineering_patterns = self._load_engineering_patterns()
        self.technical_vocabulary = self._load_technical_vocabulary()
    
    def _load_engineering_patterns(self) -> Dict[str, str]:
        """Load engineering-specific patterns."""
        return {
            "specifications": r'([A-Za-z\s]+):\s*(\d+(?:\.\d+)?)\s*(mm|cm|m|in|ft|MPa|GPa|psi|ksi|N|kN|MN|kg|g|V|A|W|kW|MW|°C|°F|K|rpm|Hz|kHz|MHz|GHz)',
            "requirements": r'(?:requirement|specification|must|shall|should)\s*[:\-]?\s*([^.!?]+)',
            "constraints": r'(?:constraint|limit|maximum|minimum|max|min)\s*[:\-]?\s*([^.!?]+)',
            "processes": r'(?:process|procedure|method|technique)\s*[:\-]?\s*([^.!?]+)',
            "standards": r'(?:standard|code|regulation|guideline)\s*[:\-]?\s*([^.!?]+)',
            "materials": r'(?:material|alloy|composite|polymer|ceramic)\s*[:\-]?\s*([^.!?]+)',
            "properties": r'(?:property|characteristic|attribute|parameter)\s*[:\-]?\s*([^.!?]+)',
            "tests": r'(?:test|testing|validation|verification)\s*[:\-]?\s*([^.!?]+)',
            "failures": r'(?:failure|fault|defect|problem|issue)\s*[:\-]?\s*([^.!?]+)',
            "solutions": r'(?:solution|fix|remedy|corrective)\s*[:\-]?\s*([^.!?]+)'
        }
    
    def _load_technical_vocabulary(self) -> Dict[str, List[str]]:
        """Load technical vocabulary by domain."""
        return {
            "structural": [
                "beam", "column", "load", "stress", "strain", "deflection", "moment", "shear",
                "bending", "torsion", "buckling", "fatigue", "creep", "yield", "ultimate"
            ],
            "mechanical": [
                "machine", "mechanism", "gear", "bearing", "motor", "engine", "transmission",
                "pump", "compressor", "turbine", "kinematics", "dynamics", "thermodynamics"
            ],
            "electrical": [
                "circuit", "voltage", "current", "power", "resistance", "capacitance", "inductance",
                "frequency", "impedance", "transformer", "generator", "motor", "converter"
            ],
            "materials": [
                "steel", "aluminum", "titanium", "concrete", "composite", "polymer", "ceramic",
                "alloy", "crystal", "grain", "phase", "microstructure", "metallurgy"
            ],
            "manufacturing": [
                "production", "manufacturing", "assembly", "machining", "welding", "casting",
                "forging", "molding", "extrusion", "heat treatment", "surface treatment"
            ]
        }
    
    def extract_engineering_specifications(self, text: str) -> List[Dict[str, Any]]:
        """Extract engineering specifications from text."""
        specifications = []
        
        # Extract specifications using pattern
        spec_pattern = self.engineering_patterns["specifications"]
        matches = re.finditer(spec_pattern, text, re.IGNORECASE)
        
        for match in matches:
            specifications.append({
                "parameter": match.group(1).strip(),
                "value": float(match.group(2)),
                "unit": match.group(3),
                "text": match.group(0),
                "position": match.start()
            })
        
        return specifications
    
    def extract_requirements(self, text: str) -> List[Dict[str, Any]]:
        """Extract requirements from text."""
        requirements = []
        
        # Extract requirements using pattern
        req_pattern = self.engineering_patterns["requirements"]
        matches = re.finditer(req_pattern, text, re.IGNORECASE)
        
        for match in matches:
            requirements.append({
                "requirement": match.group(1).strip(),
                "text": match.group(0),
                "position": match.start(),
                "type": "requirement"
            })
        
        return requirements
    
    def extract_constraints(self, text: str) -> List[Dict[str, Any]]:
        """Extract constraints from text."""
        constraints = []
        
        # Extract constraints using pattern
        constraint_pattern = self.engineering_patterns["constraints"]
        matches = re.finditer(constraint_pattern, text, re.IGNORECASE)
        
        for match in matches:
            constraints.append({
                "constraint": match.group(1).strip(),
                "text": match.group(0),
                "position": match.start(),
                "type": "constraint"
            })
        
        return constraints
    
    def extract_processes(self, text: str) -> List[Dict[str, Any]]:
        """Extract processes from text."""
        processes = []
        
        # Extract processes using pattern
        process_pattern = self.engineering_patterns["processes"]
        matches = re.finditer(process_pattern, text, re.IGNORECASE)
        
        for match in matches:
            processes.append({
                "process": match.group(1).strip(),
                "text": match.group(0),
                "position": match.start(),
                "type": "process"
            })
        
        return processes
    
    def extract_engineering_domains(self, text: str) -> Dict[str, float]:
        """Extract engineering domains and their relevance scores."""
        domain_scores = {}
        text_lower = text.lower()
        
        for domain, vocabulary in self.technical_vocabulary.items():
            score = 0
            for term in vocabulary:
                if term in text_lower:
                    score += 1
            
            if score > 0:
                domain_scores[domain] = score / len(vocabulary)
        
        return domain_scores
    
    def extract_technical_relationships(self, text: str) -> List[Dict[str, Any]]:
        """Extract technical relationships from text."""
        relationships = []
        
        # Engineering-specific relationship patterns
        tech_patterns = {
            "affects": r'([A-Za-z\s]+)\s+affects?\s+([A-Za-z\s]+)',
            "influences": r'([A-Za-z\s]+)\s+influences?\s+([A-Za-z\s]+)',
            "depends_on": r'([A-Za-z\s]+)\s+depends?\s+on\s+([A-Za-z\s]+)',
            "causes": r'([A-Za-z\s]+)\s+causes?\s+([A-Za-z\s]+)',
            "prevents": r'([A-Za-z\s]+)\s+prevents?\s+([A-Za-z\s]+)',
            "improves": r'([A-Za-z\s]+)\s+improves?\s+([A-Za-z\s]+)',
            "reduces": r'([A-Za-z\s]+)\s+reduces?\s+([A-Za-z\s]+)',
            "increases": r'([A-Za-z\s]+)\s+increases?\s+([A-Za-z\s]+)'
        }
        
        for predicate, pattern in tech_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            for subject, object_ in matches:
                relationships.append({
                    "subject": subject.strip(),
                    "predicate": predicate,
                    "object": object_.strip(),
                    "confidence": 0.8,
                    "type": "technical"
                })
        
        return relationships
    
    def extract_failure_modes(self, text: str) -> List[Dict[str, Any]]:
        """Extract failure modes from text."""
        failure_modes = []
        
        # Common failure mode patterns
        failure_patterns = [
            r'(?:failure|fault|defect|problem|issue)\s*[:\-]?\s*([^.!?]+)',
            r'(?:crack|cracking|fracture|breaking|rupture)\s*[:\-]?\s*([^.!?]+)',
            r'(?:corrosion|rust|oxidation)\s*[:\-]?\s*([^.!?]+)',
            r'(?:fatigue|wear|degradation)\s*[:\-]?\s*([^.!?]+)',
            r'(?:deformation|distortion|buckling)\s*[:\-]?\s*([^.!?]+)'
        ]
        
        for pattern in failure_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                failure_modes.append({
                    "failure_mode": match.group(1).strip(),
                    "text": match.group(0),
                    "position": match.start(),
                    "type": "failure_mode"
                })
        
        return failure_modes
    
    def extract_solutions(self, text: str) -> List[Dict[str, Any]]:
        """Extract solutions from text."""
        solutions = []
        
        # Solution patterns
        solution_patterns = [
            r'(?:solution|fix|remedy|corrective)\s*[:\-]?\s*([^.!?]+)',
            r'(?:recommendation|suggestion|proposal)\s*[:\-]?\s*([^.!?]+)',
            r'(?:improvement|enhancement|optimization)\s*[:\-]?\s*([^.!?]+)',
            r'(?:prevention|mitigation|reduction)\s*[:\-]?\s*([^.!?]+)'
        ]
        
        for pattern in solution_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                solutions.append({
                    "solution": match.group(1).strip(),
                    "text": match.group(0),
                    "position": match.start(),
                    "type": "solution"
                })
        
        return solutions
    
    def extract_comprehensive_knowledge(self, text: str) -> Dict[str, Any]:
        """Extract comprehensive engineering knowledge from text."""
        knowledge = {
            "specifications": self.extract_engineering_specifications(text),
            "requirements": self.extract_requirements(text),
            "constraints": self.extract_constraints(text),
            "processes": self.extract_processes(text),
            "relationships": self.extract_technical_relationships(text),
            "failure_modes": self.extract_failure_modes(text),
            "solutions": self.extract_solutions(text),
            "domains": self.extract_engineering_domains(text),
            "entities": self.extract_entities(text),
            "concepts": self.extract_concepts(text)
        }
        
        # Build knowledge graph
        all_entities = knowledge["entities"] + knowledge["specifications"]
        all_relationships = knowledge["relationships"]
        self.build_knowledge_graph(all_entities, all_relationships)
        
        return knowledge
    
    def analyze_knowledge_coherence(self, knowledge: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze coherence of extracted knowledge."""
        coherence_analysis = {
            "completeness_score": 0.0,
            "consistency_score": 0.0,
            "coherence_score": 0.0,
            "issues": []
        }
        
        # Check completeness
        required_elements = ["specifications", "requirements", "entities"]
        present_elements = [key for key in required_elements if knowledge.get(key)]
        completeness_score = len(present_elements) / len(required_elements)
        coherence_analysis["completeness_score"] = completeness_score
        
        # Check consistency
        specifications = knowledge.get("specifications", [])
        requirements = knowledge.get("requirements", [])
        
        if specifications and requirements:
            # Check if specifications align with requirements
            consistency_score = 0.8  # Placeholder - would need domain-specific logic
            coherence_analysis["consistency_score"] = consistency_score
        else:
            coherence_analysis["issues"].append("Missing specifications or requirements for consistency check")
        
        # Overall coherence score
        coherence_analysis["coherence_score"] = (
            coherence_analysis["completeness_score"] + 
            coherence_analysis["consistency_score"]
        ) / 2
        
        return coherence_analysis
    
    def get_status(self) -> Dict[str, Any]:
        """Get extractor status."""
        base_status = super().get_status()
        base_status.update({
            "engineering_patterns_count": len(self.engineering_patterns),
            "technical_vocabulary_domains": list(self.technical_vocabulary.keys())
        })
        return base_status
