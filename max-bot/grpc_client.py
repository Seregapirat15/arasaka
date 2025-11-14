"""
gRPC client for MAX bot to communicate with ML service
"""
import grpc
import logging
from typing import List, Optional
import sys
import os

# Add src directory to path for gRPC proto imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from infrastructure.grpc import arasaka_pb2, arasaka_pb2_grpc
from config import settings

logger = logging.getLogger(__name__)


class MLServiceClient:
    """
    gRPC client for communicating with Arasaka ML service
    """
    
    def __init__(self, host: str = "localhost", port: int = 8001, timeout: int = 30):
        """
        Initialize gRPC client
        
        Args:
            host: ML service host
            port: ML service port
            timeout: Request timeout in seconds
        """
        self.address = f"{host}:{port}"
        self.timeout = timeout
        self._channel = None
        self._stub = None
        
        logger.info(f"ML Service client configured for {self.address}")
    
    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
    
    def connect(self):
        """Establish connection to gRPC server"""
        if self._channel is None:
            self._channel = grpc.insecure_channel(self.address)
            self._stub = arasaka_pb2_grpc.ArasakaServiceStub(self._channel)
            logger.info(f"Connected to ML service at {self.address}")
    
    def close(self):
        """Close gRPC channel"""
        if self._channel:
            self._channel.close()
            self._channel = None
            self._stub = None
            logger.debug("Closed connection to ML service")
    
    def health_check(self) -> bool:
        """
        Check if ML service is healthy
        
        Returns:
            True if service is healthy, False otherwise
        """
        try:
            self.connect()
            request = arasaka_pb2.HealthRequest()
            response = self._stub.HealthCheck(request, timeout=5)
            is_healthy = response.status == "healthy"
            
            if is_healthy:
                logger.info("ML service is healthy")
            else:
                logger.warning(f"ML service status: {response.status}")
            
            return is_healthy
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def search_answers(
        self, 
        query: str, 
        limit: int = 5, 
        score_threshold: float = 0.5
    ) -> List[dict]:
        """
        Search for answers using ML service
        
        Args:
            query: Question text
            limit: Maximum number of results
            score_threshold: Minimum similarity score
        
        Returns:
            List of search results with answers and scores
        """
        try:
            self.connect()
            
            # Create request
            request = arasaka_pb2.SearchRequest(
                query=query,
                limit=limit,
                score_threshold=score_threshold
            )
            
            # Call gRPC service
            response = self._stub.SearchAnswers(request, timeout=self.timeout)
            
            # Parse results
            results = []
            for result in response.results:
                results.append({
                    'answer': {
                        'id': result.id,
                        'text': result.answer,
                        'answer_id': result.answer_id
                    },
                    'score': result.score
                })
            
            logger.info(f"Found {len(results)} answers for query: {query[:50]}...")
            return results
            
        except grpc.RpcError as e:
            logger.error(f"gRPC error during search: {e.code()} - {e.details()}")
            raise Exception(f"ML service error: {e.details()}")
        except Exception as e:
            logger.error(f"Error during search: {e}", exc_info=True)
            raise


# Singleton instance
_ml_client = None


def get_ml_client() -> MLServiceClient:
    """
    Get singleton ML service client
    
    Returns:
        MLServiceClient instance
    """
    global _ml_client
    if _ml_client is None:
        _ml_client = MLServiceClient(
            host=settings.ml_service_host,
            port=settings.ml_service_port,
            timeout=settings.ml_service_timeout
        )
    return _ml_client

