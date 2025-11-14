"""
Domain service for question operations
"""
from abc import ABC, abstractmethod
from typing import List
from ..entities.answer import Answer


class QuestionService(ABC):   
    @abstractmethod
    async def find_similar_answers(
        self, 
        query: str,
        limit: int = None,
        score_threshold: float = None
    ) -> List[Answer]:
        """
        Find similar answers for a given query
        
        Args:
            query: User's question
            limit: Maximum number of results
            score_threshold: Minimum similarity score
            
        Returns:
            List of similar answers as domain entities
        """
        pass
