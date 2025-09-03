"""
Text processing utilities for engineering applications.
"""

import re
import string
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tag import pos_tag
import spacy


class TextProcessor:
    """
    Basic text processing utilities.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
        
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            nltk.download('averaged_perceptron_tagger')
        
        self.stop_words = set(stopwords.words('english'))
    
    def preprocess(self, text: str) -> str:
        """
        Basic text preprocessing.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep alphanumeric and basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:]', ' ', text)
        
        # Remove extra spaces
        text = text.strip()
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        return word_tokenize(text)
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stopwords from token list."""
        return [token for token in tokens if token not in self.stop_words]
    
    def stem_tokens(self, tokens: List[str]) -> List[str]:
        """Stem tokens."""
        return [self.stemmer.stem(token) for token in tokens]
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """Lemmatize tokens."""
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def extract_features(self, text: str) -> Dict[str, Any]:
        """Extract basic text features."""
        tokens = self.tokenize(text)
        sentences = sent_tokenize(text)
        
        return {
            "word_count": len(tokens),
            "sentence_count": len(sentences),
            "avg_word_length": np.mean([len(word) for word in tokens]) if tokens else 0,
            "avg_sentence_length": np.mean([len(sent.split()) for sent in sentences]) if sentences else 0,
            "unique_words": len(set(tokens)),
            "vocabulary_richness": len(set(tokens)) / len(tokens) if tokens else 0,
            "punctuation_count": sum(1 for char in text if char in string.punctuation),
            "digit_count": sum(1 for char in text if char.isdigit()),
            "uppercase_count": sum(1 for char in text if char.isupper())
        }


class EngineeringTextProcessor(TextProcessor):
    """
    Specialized text processor for engineering applications.
    """
    
    def __init__(self):
        super().__init__()
        
        # Engineering-specific stopwords
        self.engineering_stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
        }
        
        # Engineering terminology patterns
        self.engineering_patterns = {
            'measurements': r'\d+(?:\.\d+)?\s*(?:mm|cm|m|in|ft|MPa|GPa|psi|ksi|N|kN|MN|kg|g|V|A|W|kW|MW|°C|°F|K|rpm|Hz|kHz|MHz|GHz)',
            'dimensions': r'\d+(?:\.\d+)?\s*[x×]\s*\d+(?:\.\d+)?(?:\s*[x×]\s*\d+(?:\.\d+)?)?',
            'percentages': r'\d+(?:\.\d+)?%',
            'ratios': r'\d+(?:\.\d+)?\s*:\s*\d+(?:\.\d+)?',
            'equations': r'[A-Za-z]\s*[=<>≤≥]\s*\d+(?:\.\d+)?',
            'references': r'\[[\d,\s]+\]|\([\d,\s]+\)',
            'standards': r'(?:ASTM|ISO|ANSI|BS|DIN|JIS)\s*[A-Z0-9\-]+',
            'materials': r'(?:steel|aluminum|titanium|concrete|composite|polymer|ceramic|alloy)',
            'processes': r'(?:welding|machining|casting|forging|heat\s+treatment|annealing)',
            'properties': r'(?:strength|stiffness|ductility|toughness|hardness|fatigue|creep)'
        }
    
    def preprocess(self, text: str) -> str:
        """
        Engineering-specific text preprocessing.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        # Basic preprocessing
        text = super().preprocess(text)
        
        # Preserve engineering terminology
        text = self._preserve_engineering_terms(text)
        
        # Normalize units and measurements
        text = self._normalize_measurements(text)
        
        # Clean up technical references
        text = self._clean_technical_references(text)
        
        return text
    
    def _preserve_engineering_terms(self, text: str) -> str:
        """Preserve important engineering terminology."""
        # Replace common abbreviations with full forms for better processing
        abbreviations = {
            'max': 'maximum',
            'min': 'minimum',
            'avg': 'average',
            'temp': 'temperature',
            'press': 'pressure',
            'vel': 'velocity',
            'accel': 'acceleration',
            'def': 'deflection',
            'stress': 'stress',
            'strain': 'strain'
        }
        
        for abbr, full in abbreviations.items():
            text = re.sub(r'\b' + abbr + r'\b', full, text, flags=re.IGNORECASE)
        
        return text
    
    def _normalize_measurements(self, text: str) -> str:
        """Normalize measurements and units."""
        # Normalize common unit variations
        unit_normalizations = {
            r'\bmpa\b': 'MPa',
            r'\bgpa\b': 'GPa',
            r'\bpsi\b': 'psi',
            r'\bksi\b': 'ksi',
            r'\bnewton\b': 'N',
            r'\bkilonewton\b': 'kN',
            r'\bmeganewton\b': 'MN',
            r'\bkilogram\b': 'kg',
            r'\bgram\b': 'g',
            r'\bmillimeter\b': 'mm',
            r'\bcentimeter\b': 'cm',
            r'\bmeter\b': 'm',
            r'\binch\b': 'in',
            r'\bfoot\b': 'ft',
            r'\bvolt\b': 'V',
            r'\bampere\b': 'A',
            r'\bwatt\b': 'W',
            r'\bkilowatt\b': 'kW',
            r'\bmegawatt\b': 'MW',
            r'\bohm\b': 'Ω',
            r'\bfarad\b': 'F',
            r'\bhenry\b': 'H',
            r'\bcelsius\b': '°C',
            r'\bfahrenheit\b': '°F',
            r'\bkelvin\b': 'K',
            r'\brevolutions?\s+per\s+minute\b': 'rpm',
            r'\bhertz\b': 'Hz',
            r'\bkilohz\b': 'kHz',
            r'\bmegahz\b': 'MHz',
            r'\bgigahz\b': 'GHz'
        }
        
        for pattern, replacement in unit_normalizations.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def _clean_technical_references(self, text: str) -> str:
        """Clean up technical references while preserving important information."""
        # Remove excessive whitespace around technical terms
        text = re.sub(r'\s+([=<>≤≥])\s+', r' \1 ', text)
        
        # Normalize mathematical operators
        text = re.sub(r'[≤]', '<=', text)
        text = re.sub(r'[≥]', '>=', text)
        
        return text
    
    def extract_engineering_features(self, text: str) -> Dict[str, Any]:
        """Extract engineering-specific features."""
        basic_features = self.extract_features(text)
        
        # Engineering-specific features
        engineering_features = {
            "measurement_count": len(re.findall(self.engineering_patterns['measurements'], text, re.IGNORECASE)),
            "dimension_count": len(re.findall(self.engineering_patterns['dimensions'], text)),
            "percentage_count": len(re.findall(self.engineering_patterns['percentages'], text)),
            "ratio_count": len(re.findall(self.engineering_patterns['ratios'], text)),
            "equation_count": len(re.findall(self.engineering_patterns['equations'], text)),
            "reference_count": len(re.findall(self.engineering_patterns['references'], text)),
            "standard_count": len(re.findall(self.engineering_patterns['standards'], text, re.IGNORECASE)),
            "material_count": len(re.findall(self.engineering_patterns['materials'], text, re.IGNORECASE)),
            "process_count": len(re.findall(self.engineering_patterns['processes'], text, re.IGNORECASE)),
            "property_count": len(re.findall(self.engineering_patterns['properties'], text, re.IGNORECASE))
        }
        
        # Calculate engineering complexity score
        engineering_features["complexity_score"] = self._calculate_engineering_complexity(engineering_features)
        
        # Combine with basic features
        all_features = {**basic_features, **engineering_features}
        
        return all_features
    
    def _calculate_engineering_complexity(self, features: Dict[str, Any]) -> float:
        """Calculate engineering complexity score."""
        weights = {
            "measurement_count": 0.2,
            "dimension_count": 0.15,
            "equation_count": 0.25,
            "standard_count": 0.2,
            "material_count": 0.1,
            "process_count": 0.1
        }
        
        complexity = 0.0
        for feature, weight in weights.items():
            count = features.get(feature, 0)
            # Normalize count (assuming max reasonable count is 20)
            normalized_count = min(count / 20.0, 1.0)
            complexity += weight * normalized_count
        
        return min(complexity, 1.0)
    
    def extract_technical_terms(self, text: str) -> List[str]:
        """Extract technical terms from text."""
        technical_terms = []
        
        # Extract measurements
        measurements = re.findall(self.engineering_patterns['measurements'], text, re.IGNORECASE)
        technical_terms.extend(measurements)
        
        # Extract standards
        standards = re.findall(self.engineering_patterns['standards'], text, re.IGNORECASE)
        technical_terms.extend(standards)
        
        # Extract materials
        materials = re.findall(self.engineering_patterns['materials'], text, re.IGNORECASE)
        technical_terms.extend(materials)
        
        # Extract processes
        processes = re.findall(self.engineering_patterns['processes'], text, re.IGNORECASE)
        technical_terms.extend(processes)
        
        # Extract properties
        properties = re.findall(self.engineering_patterns['properties'], text, re.IGNORECASE)
        technical_terms.extend(properties)
        
        return list(set(technical_terms))  # Remove duplicates
    
    def identify_engineering_domains(self, text: str) -> List[str]:
        """Identify engineering domains mentioned in text."""
        domain_keywords = {
            "structural": ["structure", "beam", "column", "load", "stress", "strain", "deflection", "moment", "shear"],
            "mechanical": ["machine", "mechanism", "gear", "bearing", "motor", "engine", "transmission", "pump"],
            "electrical": ["circuit", "voltage", "current", "power", "electrical", "electronic", "resistor", "capacitor"],
            "civil": ["concrete", "steel", "construction", "building", "bridge", "foundation", "soil", "asphalt"],
            "materials": ["material", "alloy", "composite", "polymer", "ceramic", "metallurgy", "crystal", "grain"],
            "manufacturing": ["production", "manufacturing", "assembly", "machining", "welding", "casting", "forging"],
            "aerospace": ["aircraft", "aerospace", "aviation", "flight", "propulsion", "aerodynamics", "turbine"],
            "automotive": ["vehicle", "automotive", "engine", "transmission", "chassis", "suspension", "brake"],
            "chemical": ["chemical", "reaction", "catalyst", "polymer", "synthesis", "distillation", "extraction"],
            "environmental": ["environment", "pollution", "waste", "treatment", "sustainability", "emission", "renewable"]
        }
        
        text_lower = text.lower()
        detected_domains = []
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                detected_domains.append(domain)
        
        return detected_domains
    
    def extract_measurements(self, text: str) -> List[Dict[str, Any]]:
        """Extract measurements with their context."""
        measurements = []
        
        # Pattern for measurements with context
        pattern = r'(\d+(?:\.\d+)?)\s*(mm|cm|m|in|ft|MPa|GPa|psi|ksi|N|kN|MN|kg|g|V|A|W|kW|MW|°C|°F|K|rpm|Hz|kHz|MHz|GHz)'
        
        matches = re.finditer(pattern, text, re.IGNORECASE)
        
        for match in matches:
            value = float(match.group(1))
            unit = match.group(2)
            
            # Get context (20 characters before and after)
            start = max(0, match.start() - 20)
            end = min(len(text), match.end() + 20)
            context = text[start:end].strip()
            
            measurements.append({
                "value": value,
                "unit": unit,
                "context": context,
                "position": match.start()
            })
        
        return measurements
    
    def get_status(self) -> Dict[str, Any]:
        """Get processor status."""
        return {
            "stemmer_available": self.stemmer is not None,
            "lemmatizer_available": self.lemmatizer is not None,
            "stop_words_count": len(self.stop_words),
            "engineering_patterns_count": len(self.engineering_patterns)
        }
