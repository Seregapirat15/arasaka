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
        query: str
    ) -> List[Answer]:
        """
        Find similar answers for a given query
        
        Args:
            query: User's question
            
        Returns:
            List of similar answers as domain entities
        """
        pass
