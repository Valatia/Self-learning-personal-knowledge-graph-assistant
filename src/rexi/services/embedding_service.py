"""
Embedding service for generating vector embeddings.
"""

from typing import List, Dict, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import logging

from rexi.config.settings import get_settings

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Service for generating and managing embeddings."""
    
    def __init__(self):
        """Initialize embedding service."""
        self.settings = get_settings()
        self.model: Optional[SentenceTransformer] = None
        self.embedding_dimension = 384  # Default for all-MiniLM-L6-v2
        self._load_model()
    
    def _load_model(self):
        """Load the embedding model."""
        try:
            self.model = SentenceTransformer(self.settings.embedding_model)
            self.embedding_dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Loaded embedding model: {self.settings.embedding_model}")
            logger.info(f"Embedding dimension: {self.embedding_dimension}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def encode_text(self, text: str) -> List[float]:
        """Encode a single text to vector."""
        if not self.model:
            self._load_model()
        
        try:
            embedding = self.model.encode(text, convert_to_tensor=False)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to encode text: {e}")
            return []
    
    def encode_texts(self, texts: List[str]) -> List[List[float]]:
        """Encode multiple texts to vectors."""
        if not self.model:
            self._load_model()
        
        try:
            embeddings = self.model.encode(texts, convert_to_tensor=False)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Failed to encode texts: {e}")
            return []
    
    def encode_batch(
        self, 
        texts: List[str], 
        batch_size: int = 32,
        show_progress: bool = False
    ) -> List[List[float]]:
        """Encode texts in batches for better memory management."""
        if not self.model:
            self._load_model()
        
        try:
            embeddings = self.model.encode(
                texts, 
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_tensor=False
            )
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Failed to encode batch: {e}")
            return []
    
    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Compute cosine similarity between two embeddings."""
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Compute cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
        except Exception as e:
            logger.error(f"Failed to compute similarity: {e}")
            return 0.0
    
    def find_most_similar(
        self, 
        query_embedding: List[float], 
        candidate_embeddings: List[List[float]],
        top_k: int = 5
    ) -> List[Dict]:
        """Find most similar embeddings to query."""
        try:
            similarities = []
            for i, embedding in enumerate(candidate_embeddings):
                similarity = self.compute_similarity(query_embedding, embedding)
                similarities.append({"index": i, "similarity": similarity})
            
            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            
            return similarities[:top_k]
        except Exception as e:
            logger.error(f"Failed to find similar embeddings: {e}")
            return []
    
    def normalize_embeddings(self, embeddings: List[List[float]]) -> List[List[float]]:
        """Normalize embeddings to unit length."""
        try:
            normalized = []
            for embedding in embeddings:
                vec = np.array(embedding)
                norm = np.linalg.norm(vec)
                if norm > 0:
                    normalized_vec = (vec / norm).tolist()
                else:
                    normalized_vec = embedding
                normalized.append(normalized_vec)
            return normalized
        except Exception as e:
            logger.error(f"Failed to normalize embeddings: {e}")
            return embeddings
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings."""
        return self.embedding_dimension
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None
    
    def unload_model(self):
        """Unload model to free memory."""
        if self.model:
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Embedding model unloaded")
