"""
Question usecase implementation
"""
from typing import List
from .i_question_usecase import IQuestionUsecase
from ..delivery.dto.dto import SearchResult
from ..services.question_service import QuestionService
from ..mappers.answer_mapper import AnswerMapper
from shared.decorators import log_method_calls


class QuestionUsecase(IQuestionUsecase):
    """
    Question usecase implementation
    
    This usecase orchestrates the question-answering business process
    by coordinating domain services and handling the conversion between
    domain entities and DTOs.
    """
    
    def __init__(self, question_service: QuestionService):
        self.question_service = question_service
    
    @log_method_calls
    async def search_answers(
        self, 
        query: str
    ) -> List[SearchResult]:
        """
        Search for similar answers
        
        Args:
            query: User's question
            
        Returns:
            List of search results as DTOs
        """
        answer_entities = await self.question_service.find_similar_answers(
            query=query
        )
        
        results = []
        for answer_entity in answer_entities:
            answer_dto = AnswerMapper.entity_to_dto(answer_entity)
            search_result = SearchResult(
                answer=answer_dto,
                score=0.0,
                query=query
            )
            results.append(search_result)
        
        return results
