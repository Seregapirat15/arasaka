"""
Repository interface for Answer domain
"""
from abc import ABC, abstractmethod
from typing import List
from ..entities.answer import Answer


class AnswerRepository(ABC):
    """Abstract repository for Answer entities"""
    
    @abstractmethod
    async def find_similar_answers(
        self, 
        query_embedding: List[float], 
        limit: int, 
        score_threshold: float
    ) -> List[Answer]:
        """Find similar answers by embedding"""
        pass
    
    @abstractmethod
    async def get_collection_info(self) -> dict:
        """Get collection information"""
        pass
