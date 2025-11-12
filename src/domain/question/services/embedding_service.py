"""
Service interface for text embedding
"""
from abc import ABC, abstractmethod
from typing import List


class EmbeddingService(ABC):
    
    @abstractmethod
    def encode_text(self, text: str) -> List[float]:
        """
        Args:
            text: Input text to encode
            
        Returns:
            Embedding vector as list of floats
            
        Raises:
            ValueError: If text is empty or invalid
        """
        pass
    
    
    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """
        Get embedding vector dimension
        
        Returns:
            Dimension of the embedding vectors
        """
        pass
