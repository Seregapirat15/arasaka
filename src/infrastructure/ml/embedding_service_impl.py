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
        
        try:
            # FRIDA may require special loading parameters
            if "FRIDA" in self.model_name:
                self.model = SentenceTransformer(
                    self.model_name, 
                    device=self.device,
                    trust_remote_code=True
                )
            else:
                self.model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(f"Model loaded successfully: {self.model_name}")
        except Exception as e:
            logger.warning(f"Failed to load model {self.model_name}: {e}")
            logger.info("Trying fallback model...")
            self.model_name = "intfloat/multilingual-e5-base"
            self.model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(f"Fallback model loaded: {self.model_name}")
    
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
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = " ".join(text.split())
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\-]', ' ', text)
        
        # Remove multiple spaces
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
            
            # FRIDA may require special encoding parameters
            if "FRIDA" in self.model_name:
                # FRIDA might need specific encode_kwargs
                embedding = self.model.encode(
                    processed_text,
                    normalize_embeddings=True
                )
            else:
                embedding = self.model.encode(processed_text)
            
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to encode text: {e}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """
        Returns:
            Dimension of the embedding vectors (1536 for FRIDA)
        """
        return 1536
