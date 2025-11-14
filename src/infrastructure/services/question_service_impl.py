"""
Implementation of question service
"""
from typing import List, Optional
from logging import getLogger

from domain.question.services.question_service import QuestionService
from domain.question.services.embedding_service import EmbeddingService
from domain.question.repositories.answer_repository import AnswerRepository
from domain.question.entities.answer import Answer

logger = getLogger(__name__)


class QuestionServiceImpl(QuestionService):
    def __init__(
        self, 
        embedding_service: EmbeddingService, 
        answer_repository: AnswerRepository,
        paraphrase_service: Optional[object] = None,
        use_paraphrasing: bool = True
    ):
        """
        Initialize question service
        
        Args:
            embedding_service: Service for creating text embeddings
            answer_repository: Repository for accessing answers
            paraphrase_service: Optional service for generating query paraphrases
            use_paraphrasing: Whether to use paraphrasing for search enhancement
        """
        self.embedding_service = embedding_service
        self.answer_repository = answer_repository
        self.paraphrase_service = paraphrase_service
        self.use_paraphrasing = use_paraphrasing and paraphrase_service is not None
        
        if self.use_paraphrasing:
            logger.info("Question service initialized with paraphrasing enabled")
        else:
            logger.info("Question service initialized with baseline search only")
    
    async def find_similar_answers(
        self, 
        query: str,
        limit: int = None,
        score_threshold: float = None
    ) -> List[Answer]:
        """
        Find similar answers for a given query using paraphrasing if enabled
        
        Args:
            query: User's question
            limit: Maximum number of results
            score_threshold: Minimum similarity score
            
        Returns:
            List of similar answers as domain entities
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        try:
            # Use paraphrasing if enabled and service is available
            if self.use_paraphrasing:
                return await self._find_with_paraphrasing(
                    query=query,
                    limit=limit,
                    score_threshold=score_threshold
                )
            else:
                # Baseline search
                return await self._baseline_search(
                    query=query,
                    limit=limit,
                    score_threshold=score_threshold
                )
            
        except Exception as e:
            logger.error(f"Failed to find similar answers: {e}")
            raise
    
    async def _baseline_search(
        self,
        query: str,
        limit: int = None,
        score_threshold: float = None
    ) -> List[Answer]:
        """
        Baseline search without paraphrasing
        """
        query_embedding = self.embedding_service.encode_text(query)
        search_results = await self.answer_repository.find_similar_answers(
            query_embedding=query_embedding,
            limit=limit,
            score_threshold=score_threshold
        )
        
        logger.info(f"Found {len(search_results)} similar answers for query: {query[:50]}...")
        return search_results
    
    async def _find_with_paraphrasing(
        self,
        query: str,
        limit: int = None,
        score_threshold: float = None
    ) -> List[Answer]:
        """
        Enhanced search using query paraphrasing and voting ranker
        
        Generates 5 paraphrases, searches top-5 for each, then ranks using weighted voting
        """
        try:
            # Import here to avoid circular dependencies
            from infrastructure.paraphrasing.paraphrase_service import ParaphraseService, SimpleParaphraseService
            from infrastructure.paraphrasing.voting_ranker import VotingRanker, AnswerResult
            
            # Generate 5 paraphrases
            num_paraphrases = 5
            paraphrases = self.paraphrase_service.generate_paraphrases(
                query,
                num_paraphrases=num_paraphrases
            )
            
            if not paraphrases:
                logger.warning(f"No paraphrases generated for query: {query[:50]}..., using baseline")
                return await self._baseline_search(query, limit, score_threshold)
            
            # If only original query returned, use baseline
            if len(paraphrases) == 1 and paraphrases[0].lower() == query.lower():
                logger.debug("Only original query returned as paraphrase, using baseline")
                return await self._baseline_search(query, limit, score_threshold)
            
            logger.debug(f"Generated {len(paraphrases)} paraphrases for query: {query[:50]}...")
            
            # Search for top-5 answers for each paraphrase
            search_results_by_paraphrase = {}
            answer_entity_map = {}  # Store original Answer entities by answer_id
            limit_per_paraphrase = 5
            
            # Use lower threshold for paraphrases to get more candidates
            paraphrase_threshold = (score_threshold * 0.7) if score_threshold else 0.2
            
            for paraphrase in paraphrases:
                query_embedding = self.embedding_service.encode_text(paraphrase)
                answers = await self.answer_repository.find_similar_answers(
                    query_embedding=query_embedding,
                    limit=limit_per_paraphrase,
                    score_threshold=paraphrase_threshold
                )
                
                # Convert Answer entities to AnswerResult for voting
                # Also store original Answer entities
                answer_results = []
                for answer in answers:
                    score = getattr(answer, 'score', 0.0)
                    answer_results.append(AnswerResult(
                        answer_id=answer.answer_id,
                        answer_text=answer.text,
                        score=float(score)
                    ))
                    # Store original Answer entity (first occurrence wins)
                    if answer.answer_id not in answer_entity_map:
                        answer_entity_map[answer.answer_id] = answer
                
                search_results_by_paraphrase[paraphrase] = answer_results
            
            # Use weighted voting to rank answers
            ranker = VotingRanker(voting_method="weighted")
            ranked_results = ranker.rank_answers(search_results_by_paraphrase)
            
            # Convert back to Answer entities using stored entities
            results = []
            seen_ids = set()
            for answer_id, answer_text, avg_score, max_score, ranking_score in ranked_results:
                if answer_id in seen_ids:
                    continue
                seen_ids.add(answer_id)
                
                # Use original answer entity if available, otherwise create new one
                if answer_id in answer_entity_map:
                    answer = answer_entity_map[answer_id]
                    # Update score to max_score from voting
                    answer.score = max_score
                else:
                    # Create new Answer entity if not found (shouldn't happen normally)
                    logger.warning(f"Answer entity not found for answer_id: {answer_id}, creating new one")
                    answer = Answer(
                        id=answer_id,
                        text=answer_text,
                        answer_id=answer_id,
                        metadata={}
                    )
                    answer.score = max_score
                
                results.append(answer)
            
            # Apply limit
            if limit:
                results = results[:limit]
            
            logger.info(
                f"Found {len(results)} similar answers using paraphrasing "
                f"(from {len(paraphrases)} paraphrases) for query: {query[:50]}..."
            )
            
            return results
            
        except ImportError as e:
            logger.warning(f"Paraphrasing modules not available: {e}, falling back to baseline")
            return await self._baseline_search(query, limit, score_threshold)
        except Exception as e:
            logger.error(f"Paraphrasing search failed: {e}, falling back to baseline", exc_info=True)
            return await self._baseline_search(query, limit, score_threshold)
