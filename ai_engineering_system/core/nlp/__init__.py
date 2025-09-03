"""
Natural Language Processing module for engineering applications.
"""

from .nlp_module import NLPModule
from .text_processor import TextProcessor, EngineeringTextProcessor
from .document_analyzer import DocumentAnalyzer, TechnicalDocumentAnalyzer
from .chatbot import EngineeringChatbot, TechnicalAssistant
from .knowledge_extractor import KnowledgeExtractor, EngineeringKnowledgeExtractor
from .embeddings import EmbeddingGenerator, EngineeringEmbeddings

__all__ = [
    "NLPModule",
    "TextProcessor",
    "EngineeringTextProcessor",
    "DocumentAnalyzer",
    "TechnicalDocumentAnalyzer",
    "EngineeringChatbot",
    "TechnicalAssistant",
    "KnowledgeExtractor",
    "EngineeringKnowledgeExtractor",
    "EmbeddingGenerator",
    "EngineeringEmbeddings",
]
