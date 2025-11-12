"""
Qdrant client for vector storage and search
"""
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, Filter, FieldCondition, MatchValue
import numpy as np
from logging import getLogger

from config.config import settings
from domain.question.entities.answer import Answer
from domain.question.repositories.answer_repository import AnswerRepository

logger = getLogger(__name__)


class QdrantRepository(AnswerRepository):
    """Service for interacting with Qdrant vector database"""
    
    def __init__(self, host: Optional[str] = None, port: Optional[int] = None):
        self.host = host or settings.qdrant_host
        self.port = port or settings.qdrant_port
        self.collection_name = settings.qdrant_collection_name
        
        self.client = QdrantClient(host=self.host, port=self.port)
        logger.info(f"Qdrant client initialized: {self.host}:{self.port}")
    
    def create_collection(self, vector_size: int, distance: Distance = Distance.COSINE) -> bool:
        """
        Create a new collection in Qdrant
        
        Args:
            vector_size: Size of the vectors
            distance: Distance metric for similarity search
            
        Returns:
            True if collection was created successfully
        """
        try:
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name in collection_names:
                logger.info(f"Collection {self.collection_name} already exists")
                return True
            
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=distance
                )
            )
            logger.info(f"Collection {self.collection_name} created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            return False
    
    
    async def find_similar_answers(self, query_embedding: List[float], 
                                  limit: int = 5,
                                  score_threshold: float = 0.7) -> List[Answer]:
        """
        Search for similar answers based on user question
        
        Args:
            query_embedding: Vector embedding of the user's question
            limit: Maximum number of results
            score_threshold: Minimum similarity score
            
        Returns:
            List of similar answers with scores
        """
        try:
            limit = limit or settings.search_limit
            score_threshold = score_threshold or settings.similarity_threshold
            
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=Filter(
                    must=[
                        FieldCondition(
                            key="is_visible",
                            match=MatchValue(value=True)
                        )
                    ]
                )
            )
            
            results = []
            for result in search_results:
                answer = Answer(
                    id=str(result.id),
                    text=result.payload.get("answer", ""),
                    answer_id=result.payload.get("answer_id", ""),
                    metadata={k: v for k, v in result.payload.items() 
                             if k not in ["question", "answer", "question_id", "is_visible"]}
                )
                answer.score = result.score
                results.append(answer)
            
            logger.debug(f"Found {len(results)} similar answers")
            return results
            
        except Exception as e:
            logger.error(f"Failed to search similar answers: {e}")
            return []
    
    
    async def get_collection_info(self) -> dict:
        """
        Get information about the collection
        
        Returns:
            Dictionary with collection information
        """
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                "name": collection_info.config.params.vectors.size,
                "vector_size": collection_info.config.params.vectors.size,
                "distance": collection_info.config.params.vectors.distance,
                "points_count": collection_info.points_count,
                "status": collection_info.status
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {}
    
