"""
Implementation of embedding service using SentenceTransformer 
"""
from typing import List
from logging import getLogger
from sentence_transformers import SentenceTransformer

from config.config import settings
from domain.question.services.embedding_service import EmbeddingService

logger = getLogger(__name__)


class EmbeddingServiceImpl(EmbeddingService):
    def __init__(self):
        self.model_name = settings.model_name
        self.device = settings.model_device
        
        logger.info(f"Loading model: {self.model_name}")
        logger.info("This may take a few minutes on first run...")
        
        self.model = SentenceTransformer(self.model_name, device=self.device)
        logger.info(f"Model loaded successfully: {self.model_name}")
    
    def _preprocess_text(self, text: str) -> str:
        """
        Enhanced text preprocessing for better embeddings
        
        Args:
            text: Input text to preprocess
            
        Returns:
            Preprocessed text
        """
        if not text or not isinstance(text, str):
            return ""
        
        import re
        
        text = text.lower()
        text = " ".join(text.split())
        text = re.sub(r'[^\w\s\.\,\!\?\-]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def encode_text(self, text: str) -> List[float]:
        """
        Encode text to embedding vector
        
        Args:
            text: Input text to encode
            
        Returns:
            Embedding vector as list of floats
            
        Raises:
            ValueError: If text is empty or invalid
        """
        if not text or not isinstance(text, str):
            raise ValueError("Text must be a non-empty string")
        
        try:
            processed_text = self._preprocess_text(text)
            embedding = self.model.encode(
                processed_text,
                normalize_embeddings=True
            )
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to encode text: {e}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """
        Get the actual embedding dimension from the model
        
        Returns:
            Dimension of the embedding vectors (1536 for FRIDA)
        """
        try:
            if hasattr(self.model, 'get_sentence_embedding_dimension'):
                return self.model.get_sentence_embedding_dimension()
            test_embedding = self.model.encode("test", convert_to_numpy=True)
            return len(test_embedding)
        except Exception as e:
            logger.warning(f"Failed to get embedding dimension, using default: {e}")
            return 1536
