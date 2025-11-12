"""
Implementation of question service
"""
from typing import List
from logging import getLogger

from domain.question.services.question_service import QuestionService
from domain.question.services.embedding_service import EmbeddingService
from domain.question.repositories.answer_repository import AnswerRepository
from domain.question.entities.answer import Answer

logger = getLogger(__name__)


class QuestionServiceImpl(QuestionService):
    def __init__(self, embedding_service: EmbeddingService, answer_repository: AnswerRepository):
        """
        Initialize question service
        
        Args:
            embedding_service: Service for creating text embeddings
            answer_repository: Repository for accessing answers
        """
        self.embedding_service = embedding_service
        self.answer_repository = answer_repository
        logger.info("Question service implementation initialized")
    
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
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        try:
            query_embedding = self.embedding_service.encode_text(query)
            search_results = await self.answer_repository.find_similar_answers(
                query_embedding=query_embedding,
                limit=5,
                score_threshold=0.3
            )
            
            logger.info(f"Found {len(search_results)} similar answers for query: {query}")
            return search_results
            
        except Exception as e:
            logger.error(f"Failed to find similar answers: {e}")
            raise
