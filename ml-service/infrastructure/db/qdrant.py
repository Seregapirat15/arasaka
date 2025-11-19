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
    
    def __init__(self, host: Optional[str] = None, port: Optional[int] = None):
        self.host = host or settings.qdrant_host
        self.port = port or settings.qdrant_port
        self.collection_name = settings.qdrant_collection_name
        
        self.client = QdrantClient(host=self.host, port=self.port)
        logger.info(f"Qdrant client initialized: {self.host}:{self.port}")
    
    def create_collection(self, vector_size: int, distance: Distance = Distance.COSINE) -> bool:
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
        try:
            limit = limit if limit is not None and limit > 0 else settings.search_limit
            if score_threshold is None:
                score_threshold = settings.similarity_threshold
            
            # Use query_points instead of search (new API)
            # query can be a vector directly, filter goes in query_filter parameter
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="is_visible",
                        match=MatchValue(value=True)
                    )
                ]
            )
            
            search_results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                query_filter=query_filter,
                limit=limit,
                score_threshold=score_threshold
            )
            
            # If no results with filter, try without filter
            if len(search_results.points) == 0:
                logger.debug("No results with is_visible filter, trying without filter")
                search_results = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_embedding,
                    limit=limit,
                    score_threshold=score_threshold
                )
            
            results = []
            for point in search_results.points:
                answer = Answer(
                    id=str(point.id),
                    text=point.payload.get("answer", ""),
                    answer_id=point.payload.get("answer_id", ""),
                    metadata={k: v for k, v in point.payload.items() 
                             if k not in ["question", "answer", "question_id", "is_visible"]}
                )
                answer.score = point.score
                results.append(answer)
            
            logger.debug(f"Found {len(results)} similar answers")
            return results
            
        except Exception as e:
            logger.error(f"Failed to search similar answers: {e}", exc_info=True)
            return []
    
    
    async def get_collection_info(self) -> dict:
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
    
