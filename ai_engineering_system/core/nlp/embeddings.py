"""
Embedding generation utilities for engineering applications.
"""

import logging
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
import pickle
import os


class EmbeddingGenerator:
    """
    Basic embedding generation utilities.
    """
    
    def __init__(self, device: str = "cpu"):
        """
        Initialize the embedding generator.
        
        Args:
            device: Device to use for computations
        """
        self.device = device
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.vectorizers = {}
        self.embeddings_cache = {}
    
    def generate_tfidf_embeddings(self, texts: List[str], max_features: int = 1000) -> np.ndarray:
        """
        Generate TF-IDF embeddings for texts.
        
        Args:
            texts: List of texts to embed
            max_features: Maximum number of features
            
        Returns:
            TF-IDF embeddings matrix
        """
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )
        
        embeddings = vectorizer.fit_transform(texts).toarray()
        
        # Store vectorizer for later use
        self.vectorizers['tfidf'] = vectorizer
        
        return embeddings
    
    def generate_word_embeddings(self, texts: List[str], model_name: str = "word2vec") -> np.ndarray:
        """
        Generate word-level embeddings for texts.
        
        Args:
            texts: List of texts to embed
            model_name: Name of the embedding model
            
        Returns:
            Word embeddings matrix
        """
        # This would be implemented with actual word embedding models
        # For now, return placeholder
        return np.random.rand(len(texts), 300)
    
    def generate_sentence_embeddings(self, texts: List[str], model_name: str = "sentence-transformers") -> np.ndarray:
        """
        Generate sentence-level embeddings for texts.
        
        Args:
            texts: List of texts to embed
            model_name: Name of the embedding model
            
        Returns:
            Sentence embeddings matrix
        """
        # This would be implemented with actual sentence embedding models
        # For now, return placeholder
        return np.random.rand(len(texts), 768)
    
    def reduce_dimensions(self, embeddings: np.ndarray, method: str = "pca", n_components: int = 50) -> np.ndarray:
        """
        Reduce dimensionality of embeddings.
        
        Args:
            embeddings: Input embeddings
            method: Dimensionality reduction method
            n_components: Number of components to keep
            
        Returns:
            Reduced embeddings
        """
        if method == "pca":
            reducer = PCA(n_components=n_components)
        elif method == "svd":
            reducer = TruncatedSVD(n_components=n_components)
        elif method == "tsne":
            reducer = TSNE(n_components=n_components, random_state=42)
        else:
            raise ValueError(f"Unknown reduction method: {method}")
        
        reduced_embeddings = reducer.fit_transform(embeddings)
        
        # Store reducer for later use
        self.models[f"{method}_reducer"] = reducer
        
        return reduced_embeddings
    
    def compute_similarity(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
        """
        Compute similarity between two sets of embeddings.
        
        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings
            
        Returns:
            Similarity matrix
        """
        from sklearn.metrics.pairwise import cosine_similarity
        return cosine_similarity(embeddings1, embeddings2)
    
    def find_similar_texts(self, query_embedding: np.ndarray, text_embeddings: np.ndarray, 
                          texts: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find most similar texts to a query.
        
        Args:
            query_embedding: Query embedding
            text_embeddings: Text embeddings to search
            texts: Original texts
            top_k: Number of similar texts to return
            
        Returns:
            List of similar texts with scores
        """
        similarities = self.compute_similarity([query_embedding], text_embeddings)[0]
        
        # Get top-k similar texts
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        similar_texts = []
        for idx in top_indices:
            similar_texts.append({
                "text": texts[idx],
                "similarity": similarities[idx],
                "index": idx
            })
        
        return similar_texts
    
    def cluster_embeddings(self, embeddings: np.ndarray, n_clusters: int = 5) -> Dict[str, Any]:
        """
        Cluster embeddings using K-means.
        
        Args:
            embeddings: Input embeddings
            n_clusters: Number of clusters
            
        Returns:
            Clustering results
        """
        from sklearn.cluster import KMeans
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        return {
            "labels": cluster_labels,
            "centers": kmeans.cluster_centers_,
            "inertia": kmeans.inertia_,
            "model": kmeans
        }
    
    def save_embeddings(self, embeddings: np.ndarray, filepath: str):
        """Save embeddings to file."""
        with open(filepath, 'wb') as f:
            pickle.dump(embeddings, f)
        self.logger.info(f"Embeddings saved to {filepath}")
    
    def load_embeddings(self, filepath: str) -> np.ndarray:
        """Load embeddings from file."""
        with open(filepath, 'rb') as f:
            embeddings = pickle.load(f)
        self.logger.info(f"Embeddings loaded from {filepath}")
        return embeddings
    
    def get_status(self) -> Dict[str, Any]:
        """Get generator status."""
        return {
            "device": self.device,
            "models_loaded": list(self.models.keys()),
            "vectorizers_loaded": list(self.vectorizers.keys()),
            "embeddings_cached": len(self.embeddings_cache)
        }


class EngineeringEmbeddings(EmbeddingGenerator):
    """
    Specialized embedding generator for engineering applications.
    """
    
    def __init__(self, device: str = "cpu"):
        super().__init__(device)
        self.engineering_models = {}
        self.technical_vocabulary = self._load_technical_vocabulary()
        self._initialize_engineering_models()
    
    def _load_technical_vocabulary(self) -> Dict[str, List[str]]:
        """Load technical vocabulary for engineering applications."""
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
    
    def _initialize_engineering_models(self):
        """Initialize engineering-specific embedding models."""
        try:
            # Initialize sentence transformer for general text
            self.engineering_models['sentence_transformer'] = SentenceTransformer(
                'all-MiniLM-L6-v2',
                device=self.device
            )
            
            # Initialize BERT for technical text
            self.engineering_models['bert_tokenizer'] = AutoTokenizer.from_pretrained('bert-base-uncased')
            self.engineering_models['bert_model'] = AutoModel.from_pretrained('bert-base-uncased')
            
            if self.device == "cuda":
                self.engineering_models['bert_model'] = self.engineering_models['bert_model'].cuda()
            
            self.logger.info("Engineering embedding models initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing engineering models: {e}")
            self.engineering_models = {}
    
    async def generate_embeddings(self, text: str, method: str = "sentence_transformer") -> np.ndarray:
        """
        Generate embeddings for engineering text.
        
        Args:
            text: Input text
            method: Embedding method to use
            
        Returns:
            Text embeddings
        """
        if method == "sentence_transformer":
            return await self._generate_sentence_transformer_embeddings(text)
        elif method == "bert":
            return await self._generate_bert_embeddings(text)
        elif method == "tfidf":
            return await self._generate_tfidf_embeddings(text)
        else:
            raise ValueError(f"Unknown embedding method: {method}")
    
    async def _generate_sentence_transformer_embeddings(self, text: str) -> np.ndarray:
        """Generate embeddings using sentence transformer."""
        if 'sentence_transformer' not in self.engineering_models:
            raise ValueError("Sentence transformer model not available")
        
        model = self.engineering_models['sentence_transformer']
        embedding = model.encode([text])
        return embedding[0]
    
    async def _generate_bert_embeddings(self, text: str) -> np.ndarray:
        """Generate embeddings using BERT."""
        if 'bert_model' not in self.engineering_models or 'bert_tokenizer' not in self.engineering_models:
            raise ValueError("BERT model not available")
        
        tokenizer = self.engineering_models['bert_tokenizer']
        model = self.engineering_models['bert_model']
        
        # Tokenize and encode
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        
        if self.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            # Use [CLS] token embedding
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return embedding[0]
    
    async def _generate_tfidf_embeddings(self, text: str) -> np.ndarray:
        """Generate TF-IDF embeddings."""
        if 'tfidf' not in self.vectorizers:
            # Initialize TF-IDF vectorizer
            self.vectorizers['tfidf'] = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95
            )
            # Fit on the single text
            self.vectorizers['tfidf'].fit([text])
        
        vectorizer = self.vectorizers['tfidf']
        embedding = vectorizer.transform([text]).toarray()
        return embedding[0]
    
    def generate_domain_specific_embeddings(self, text: str, domain: str) -> np.ndarray:
        """
        Generate domain-specific embeddings for engineering text.
        
        Args:
            text: Input text
            domain: Engineering domain
            
        Returns:
            Domain-specific embeddings
        """
        # Get domain vocabulary
        domain_vocabulary = self.technical_vocabulary.get(domain, [])
        
        # Create domain-specific TF-IDF vectorizer
        if domain not in self.vectorizers:
            self.vectorizers[domain] = TfidfVectorizer(
                vocabulary=domain_vocabulary,
                ngram_range=(1, 2),
                min_df=1
            )
        
        vectorizer = self.vectorizers[domain]
        
        # Fit and transform
        if not hasattr(vectorizer, 'vocabulary_'):
            vectorizer.fit([text])
        
        embedding = vectorizer.transform([text]).toarray()
        return embedding[0]
    
    def generate_technical_embeddings(self, texts: List[str], method: str = "sentence_transformer") -> np.ndarray:
        """
        Generate embeddings for multiple technical texts.
        
        Args:
            texts: List of technical texts
            method: Embedding method
            
        Returns:
            Embeddings matrix
        """
        embeddings = []
        
        for text in texts:
            if method == "sentence_transformer":
                embedding = self._generate_sentence_transformer_embeddings(text)
            elif method == "bert":
                embedding = self._generate_bert_embeddings(text)
            elif method == "tfidf":
                embedding = self._generate_tfidf_embeddings(text)
            else:
                raise ValueError(f"Unknown embedding method: {method}")
            
            embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def compute_engineering_similarity(self, text1: str, text2: str, method: str = "sentence_transformer") -> float:
        """
        Compute similarity between two engineering texts.
        
        Args:
            text1: First text
            text2: Second text
            method: Embedding method
            
        Returns:
            Similarity score
        """
        embedding1 = self.generate_embeddings(text1, method)
        embedding2 = self.generate_embeddings(text2, method)
        
        # Compute cosine similarity
        similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        return float(similarity)
    
    def find_similar_engineering_texts(self, query: str, texts: List[str], 
                                     method: str = "sentence_transformer", top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find similar engineering texts to a query.
        
        Args:
            query: Query text
            texts: Texts to search
            method: Embedding method
            top_k: Number of similar texts to return
            
        Returns:
            List of similar texts with scores
        """
        # Generate embeddings
        query_embedding = self.generate_embeddings(query, method)
        text_embeddings = self.generate_technical_embeddings(texts, method)
        
        # Find similar texts
        return self.find_similar_texts(query_embedding, text_embeddings, texts, top_k)
    
    def cluster_engineering_texts(self, texts: List[str], n_clusters: int = 5, 
                                method: str = "sentence_transformer") -> Dict[str, Any]:
        """
        Cluster engineering texts by similarity.
        
        Args:
            texts: Texts to cluster
            n_clusters: Number of clusters
            method: Embedding method
            
        Returns:
            Clustering results
        """
        # Generate embeddings
        embeddings = self.generate_technical_embeddings(texts, method)
        
        # Cluster embeddings
        clustering_results = self.cluster_embeddings(embeddings, n_clusters)
        
        # Add text information to results
        clustering_results["texts"] = texts
        clustering_results["clusters"] = {}
        
        for i, label in enumerate(clustering_results["labels"]):
            if label not in clustering_results["clusters"]:
                clustering_results["clusters"][label] = []
            clustering_results["clusters"][label].append({
                "text": texts[i],
                "index": i
            })
        
        return clustering_results
    
    def analyze_engineering_domains(self, texts: List[str]) -> Dict[str, Any]:
        """
        Analyze engineering domains in texts using embeddings.
        
        Args:
            texts: Texts to analyze
            
        Returns:
            Domain analysis results
        """
        domain_analysis = {}
        
        for domain in self.technical_vocabulary.keys():
            # Generate domain-specific embeddings
            domain_embeddings = []
            for text in texts:
                embedding = self.generate_domain_specific_embeddings(text, domain)
                domain_embeddings.append(embedding)
            
            domain_embeddings = np.array(domain_embeddings)
            
            # Calculate domain relevance scores
            relevance_scores = np.sum(domain_embeddings, axis=1)
            
            domain_analysis[domain] = {
                "relevance_scores": relevance_scores.tolist(),
                "average_relevance": float(np.mean(relevance_scores)),
                "max_relevance": float(np.max(relevance_scores)),
                "min_relevance": float(np.min(relevance_scores))
            }
        
        return domain_analysis
    
    def get_status(self) -> Dict[str, Any]:
        """Get generator status."""
        base_status = super().get_status()
        base_status.update({
            "engineering_models_loaded": list(self.engineering_models.keys()),
            "technical_vocabulary_domains": list(self.technical_vocabulary.keys()),
            "device": self.device
        })
        return base_status
