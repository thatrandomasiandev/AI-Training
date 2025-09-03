"""
Advanced Natural Language Processing module for engineering applications.
"""

import asyncio
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
import torch
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    pipeline, BertTokenizer, BertModel
)
import spacy
from sentence_transformers import SentenceTransformer
import openai
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from .text_processor import EngineeringTextProcessor
from .document_analyzer import TechnicalDocumentAnalyzer
from .chatbot import EngineeringChatbot
from .knowledge_extractor import EngineeringKnowledgeExtractor
from .embeddings import EngineeringEmbeddings
from ..utils.config import Config


class NLPModule:
    """
    Advanced NLP module for engineering applications.
    
    Provides comprehensive NLP capabilities including:
    - Technical document analysis
    - Engineering text processing
    - Knowledge extraction
    - Engineering chatbot
    - Semantic embeddings
    - Text classification and generation
    """
    
    def __init__(self, config: Config, device: str = "cpu"):
        """
        Initialize the NLP module.
        
        Args:
            config: Configuration object
            device: Device to use for computations
        """
        self.config = config
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.text_processor = EngineeringTextProcessor()
        self.document_analyzer = TechnicalDocumentAnalyzer()
        self.chatbot = EngineeringChatbot(config)
        self.knowledge_extractor = EngineeringKnowledgeExtractor()
        self.embeddings = EngineeringEmbeddings(device)
        
        # Load models
        self._load_models()
        
        self.logger.info("NLP Module initialized")
    
    def _load_models(self):
        """Load NLP models and tokenizers."""
        try:
            # Load spaCy model for text processing
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                self.logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
                self.nlp = None
            
            # Load BERT model for embeddings and classification
            model_name = self.config.get_nlp_config().model_name
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.bert_model = AutoModel.from_pretrained(model_name)
            
            if self.device == "cuda":
                self.bert_model = self.bert_model.cuda()
            
            # Load sentence transformer for embeddings
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Load classification pipeline
            self.classifier = pipeline(
                "text-classification",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=0 if self.device == "cuda" else -1
            )
            
            # Load summarization pipeline
            self.summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=0 if self.device == "cuda" else -1
            )
            
            # Load question answering pipeline
            self.qa_pipeline = pipeline(
                "question-answering",
                model="distilbert-base-cased-distilled-squad",
                device=0 if self.device == "cuda" else -1
            )
            
            self.logger.info("NLP models loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading NLP models: {e}")
            # Initialize with basic functionality
            self.nlp = None
            self.tokenizer = None
            self.bert_model = None
            self.sentence_model = None
            self.classifier = None
            self.summarizer = None
            self.qa_pipeline = None
    
    async def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze text using multiple NLP techniques.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Analysis results
        """
        self.logger.info(f"Analyzing text of length {len(text)}")
        
        # Basic text processing
        processed_text = self.text_processor.preprocess(text)
        
        # Extract features
        features = self.text_processor.extract_features(processed_text)
        
        # Generate embeddings
        embeddings = await self.embeddings.generate_embeddings(text)
        
        # Classify text
        classification = await self._classify_text(text)
        
        # Extract entities
        entities = self._extract_entities(processed_text)
        
        # Extract key phrases
        key_phrases = self._extract_key_phrases(processed_text)
        
        # Sentiment analysis
        sentiment = await self._analyze_sentiment(text)
        
        # Engineering-specific analysis
        engineering_analysis = self._analyze_engineering_content(text)
        
        return {
            "text_length": len(text),
            "processed_text": processed_text,
            "features": features,
            "embeddings": embeddings,
            "classification": classification,
            "entities": entities,
            "key_phrases": key_phrases,
            "sentiment": sentiment,
            "engineering_analysis": engineering_analysis,
            "confidence": self._calculate_confidence(classification, sentiment, engineering_analysis)
        }
    
    async def _classify_text(self, text: str) -> Dict[str, Any]:
        """Classify text using pre-trained models."""
        if self.classifier is None:
            return {"error": "Classifier not available"}
        
        try:
            # Truncate text if too long
            max_length = 512
            if len(text) > max_length:
                text = text[:max_length]
            
            result = self.classifier(text)
            return {
                "label": result[0]["label"],
                "score": result[0]["score"],
                "confidence": result[0]["score"]
            }
        except Exception as e:
            self.logger.error(f"Error in text classification: {e}")
            return {"error": str(e)}
    
    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities from text."""
        if self.nlp is None:
            return []
        
        try:
            doc = self.nlp(text)
            entities = []
            
            for ent in doc.ents:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "description": spacy.explain(ent.label_)
                })
            
            return entities
        except Exception as e:
            self.logger.error(f"Error extracting entities: {e}")
            return []
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from text."""
        if self.nlp is None:
            return []
        
        try:
            doc = self.nlp(text)
            key_phrases = []
            
            # Extract noun phrases
            for chunk in doc.noun_chunks:
                if len(chunk.text.split()) >= 2:  # Multi-word phrases
                    key_phrases.append(chunk.text)
            
            # Extract named entities as key phrases
            for ent in doc.ents:
                if ent.label_ in ["PERSON", "ORG", "GPE", "PRODUCT", "EVENT"]:
                    key_phrases.append(ent.text)
            
            return list(set(key_phrases))  # Remove duplicates
        except Exception as e:
            self.logger.error(f"Error extracting key phrases: {e}")
            return []
    
    async def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of text."""
        if self.classifier is None:
            return {"error": "Sentiment analyzer not available"}
        
        try:
            result = self.classifier(text)
            sentiment_score = result[0]["score"]
            sentiment_label = result[0]["label"]
            
            # Convert to engineering context
            if sentiment_label == "POSITIVE":
                engineering_sentiment = "favorable"
            elif sentiment_label == "NEGATIVE":
                engineering_sentiment = "concerning"
            else:
                engineering_sentiment = "neutral"
            
            return {
                "label": sentiment_label,
                "score": sentiment_score,
                "engineering_sentiment": engineering_sentiment
            }
        except Exception as e:
            self.logger.error(f"Error in sentiment analysis: {e}")
            return {"error": str(e)}
    
    def _analyze_engineering_content(self, text: str) -> Dict[str, Any]:
        """Analyze engineering-specific content in text."""
        engineering_keywords = [
            "design", "analysis", "stress", "strain", "load", "force", "moment",
            "material", "steel", "concrete", "aluminum", "titanium", "composite",
            "structure", "beam", "column", "joint", "connection", "welding",
            "manufacturing", "production", "quality", "testing", "inspection",
            "safety", "failure", "fatigue", "corrosion", "maintenance",
            "optimization", "efficiency", "performance", "reliability"
        ]
        
        text_lower = text.lower()
        found_keywords = [keyword for keyword in engineering_keywords if keyword in text_lower]
        
        # Calculate engineering relevance score
        relevance_score = len(found_keywords) / len(engineering_keywords)
        
        # Detect engineering domains
        domains = self._detect_engineering_domains(text)
        
        # Extract technical specifications
        specifications = self._extract_technical_specifications(text)
        
        return {
            "engineering_keywords": found_keywords,
            "relevance_score": relevance_score,
            "domains": domains,
            "specifications": specifications,
            "is_technical": relevance_score > 0.1
        }
    
    def _detect_engineering_domains(self, text: str) -> List[str]:
        """Detect engineering domains mentioned in text."""
        domains = {
            "structural": ["structure", "beam", "column", "load", "stress", "strain"],
            "mechanical": ["machine", "mechanism", "gear", "bearing", "motor"],
            "electrical": ["circuit", "voltage", "current", "power", "electrical"],
            "civil": ["concrete", "steel", "construction", "building", "bridge"],
            "materials": ["material", "alloy", "composite", "polymer", "ceramic"],
            "manufacturing": ["production", "manufacturing", "assembly", "machining"],
            "aerospace": ["aircraft", "aerospace", "aviation", "flight", "propulsion"],
            "automotive": ["vehicle", "automotive", "engine", "transmission", "chassis"]
        }
        
        text_lower = text.lower()
        detected_domains = []
        
        for domain, keywords in domains.items():
            if any(keyword in text_lower for keyword in keywords):
                detected_domains.append(domain)
        
        return detected_domains
    
    def _extract_technical_specifications(self, text: str) -> List[Dict[str, Any]]:
        """Extract technical specifications from text."""
        import re
        
        specifications = []
        
        # Extract numerical values with units
        patterns = [
            r'(\d+(?:\.\d+)?)\s*(MPa|GPa|psi|ksi|N|kN|MN|kg|g|mm|cm|m|in|ft)',
            r'(\d+(?:\.\d+)?)\s*(°C|°F|K|rpm|Hz|kHz|MHz|GHz)',
            r'(\d+(?:\.\d+)?)\s*(V|A|W|kW|MW|Ω|F|H)',
            r'(\d+(?:\.\d+)?)\s*(%|percent)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                specifications.append({
                    "value": float(match[0]),
                    "unit": match[1],
                    "type": "measurement"
                })
        
        return specifications
    
    def _calculate_confidence(self, classification: Dict, sentiment: Dict, engineering: Dict) -> float:
        """Calculate overall confidence score."""
        confidence_scores = []
        
        if "confidence" in classification:
            confidence_scores.append(classification["confidence"])
        
        if "score" in sentiment:
            confidence_scores.append(sentiment["score"])
        
        if "relevance_score" in engineering:
            confidence_scores.append(engineering["relevance_score"])
        
        return np.mean(confidence_scores) if confidence_scores else 0.0
    
    async def process_documents(self, document_paths: List[str]) -> Dict[str, Any]:
        """
        Process multiple technical documents.
        
        Args:
            document_paths: List of paths to documents
            
        Returns:
            Processing results
        """
        self.logger.info(f"Processing {len(document_paths)} documents")
        
        results = []
        for doc_path in document_paths:
            try:
                # Extract text from document
                text = self.document_analyzer.extract_text(doc_path)
                
                # Analyze the text
                analysis = await self.analyze_text(text)
                
                results.append({
                    "document": doc_path,
                    "analysis": analysis,
                    "status": "success"
                })
            except Exception as e:
                self.logger.error(f"Error processing document {doc_path}: {e}")
                results.append({
                    "document": doc_path,
                    "error": str(e),
                    "status": "failed"
                })
        
        return {
            "documents_processed": len(results),
            "successful": len([r for r in results if r["status"] == "success"]),
            "failed": len([r for r in results if r["status"] == "failed"]),
            "results": results
        }
    
    def extract_engineering_insights(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Extract engineering insights from text analysis."""
        insights = {
            "technical_complexity": self._assess_technical_complexity(analysis),
            "domain_expertise": analysis.get("engineering_analysis", {}).get("domains", []),
            "key_findings": self._extract_key_findings(analysis),
            "recommendations": self._generate_recommendations(analysis),
            "risk_assessment": self._assess_risks(analysis)
        }
        
        return insights
    
    def _assess_technical_complexity(self, analysis: Dict[str, Any]) -> str:
        """Assess technical complexity of the content."""
        engineering_analysis = analysis.get("engineering_analysis", {})
        relevance_score = engineering_analysis.get("relevance_score", 0.0)
        specifications = engineering_analysis.get("specifications", [])
        
        if relevance_score > 0.3 and len(specifications) > 5:
            return "high"
        elif relevance_score > 0.1 and len(specifications) > 2:
            return "medium"
        else:
            return "low"
    
    def _extract_key_findings(self, analysis: Dict[str, Any]) -> List[str]:
        """Extract key findings from analysis."""
        findings = []
        
        # Extract from key phrases
        key_phrases = analysis.get("key_phrases", [])
        if key_phrases:
            findings.extend(key_phrases[:5])  # Top 5 key phrases
        
        # Extract from entities
        entities = analysis.get("entities", [])
        important_entities = [e["text"] for e in entities if e["label"] in ["PRODUCT", "ORG", "PERSON"]]
        if important_entities:
            findings.extend(important_entities[:3])  # Top 3 important entities
        
        return findings
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        engineering_analysis = analysis.get("engineering_analysis", {})
        sentiment = analysis.get("sentiment", {})
        
        # Based on sentiment
        if sentiment.get("engineering_sentiment") == "concerning":
            recommendations.append("Review and address concerning aspects mentioned in the document")
        
        # Based on technical complexity
        complexity = self._assess_technical_complexity(analysis)
        if complexity == "high":
            recommendations.append("Consider expert review due to high technical complexity")
        
        # Based on domains
        domains = engineering_analysis.get("domains", [])
        if len(domains) > 1:
            recommendations.append("Multi-domain content detected - ensure cross-domain expertise")
        
        return recommendations
    
    def _assess_risks(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risks based on content analysis."""
        risks = {
            "level": "low",
            "factors": [],
            "mitigation": []
        }
        
        sentiment = analysis.get("sentiment", {})
        engineering_analysis = analysis.get("engineering_analysis", {})
        
        # Risk factors
        if sentiment.get("engineering_sentiment") == "concerning":
            risks["factors"].append("Negative sentiment detected")
            risks["level"] = "medium"
        
        if engineering_analysis.get("relevance_score", 0) < 0.05:
            risks["factors"].append("Low engineering relevance")
        
        # Mitigation strategies
        if risks["level"] == "medium":
            risks["mitigation"].append("Conduct detailed review")
        
        return risks
    
    def synthesize_insights(self, document_insights: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Synthesize insights across multiple documents."""
        all_domains = []
        all_findings = []
        all_recommendations = []
        risk_levels = []
        
        for doc_insight in document_insights:
            insights = doc_insight.get("insights", {})
            
            all_domains.extend(insights.get("domain_expertise", []))
            all_findings.extend(insights.get("key_findings", []))
            all_recommendations.extend(insights.get("recommendations", []))
            
            risk_assessment = insights.get("risk_assessment", {})
            risk_levels.append(risk_assessment.get("level", "low"))
        
        # Synthesize results
        unique_domains = list(set(all_domains))
        common_findings = self._find_common_elements(all_findings)
        priority_recommendations = self._prioritize_recommendations(all_recommendations)
        overall_risk = self._assess_overall_risk(risk_levels)
        
        return {
            "synthesized_domains": unique_domains,
            "common_findings": common_findings,
            "priority_recommendations": priority_recommendations,
            "overall_risk_assessment": overall_risk,
            "document_count": len(document_insights)
        }
    
    def _find_common_elements(self, elements: List[str]) -> List[str]:
        """Find common elements across lists."""
        from collections import Counter
        counter = Counter(elements)
        return [element for element, count in counter.most_common(5) if count > 1]
    
    def _prioritize_recommendations(self, recommendations: List[str]) -> List[str]:
        """Prioritize recommendations based on frequency and importance."""
        from collections import Counter
        counter = Counter(recommendations)
        return [rec for rec, count in counter.most_common(10)]
    
    def _assess_overall_risk(self, risk_levels: List[str]) -> str:
        """Assess overall risk level."""
        if "high" in risk_levels:
            return "high"
        elif "medium" in risk_levels:
            return "medium"
        else:
            return "low"
    
    async def chat(self, message: str, context: Optional[str] = None) -> str:
        """Chat with the engineering assistant."""
        return await self.chatbot.chat(message, context)
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the NLP module."""
        return {
            "device": self.device,
            "models_loaded": {
                "spacy": self.nlp is not None,
                "bert": self.bert_model is not None,
                "sentence_transformer": self.sentence_model is not None,
                "classifier": self.classifier is not None,
                "summarizer": self.summarizer is not None,
                "qa_pipeline": self.qa_pipeline is not None
            },
            "components": {
                "text_processor": self.text_processor.get_status(),
                "document_analyzer": self.document_analyzer.get_status(),
                "chatbot": self.chatbot.get_status(),
                "knowledge_extractor": self.knowledge_extractor.get_status(),
                "embeddings": self.embeddings.get_status()
            }
        }
    
    def cleanup(self):
        """Cleanup resources."""
        if self.bert_model is not None:
            del self.bert_model
        if self.sentence_model is not None:
            del self.sentence_model
        if self.classifier is not None:
            del self.classifier
        if self.summarizer is not None:
            del self.summarizer
        if self.qa_pipeline is not None:
            del self.qa_pipeline
        
        self.logger.info("NLP Module cleanup complete")
