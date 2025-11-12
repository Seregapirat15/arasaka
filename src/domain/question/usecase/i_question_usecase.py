"""
Interface for question usecase
"""
from abc import ABC, abstractmethod
from typing import List
from ..delivery.dto.dto import Answer, SearchResult


class IQuestionUsecase(ABC):
    """Interface for question usecase"""
    
    @abstractmethod
    async def search_answers(
        self, 
        query: str
    ) -> List[SearchResult]:
        """Search for similar answers"""
        pass
