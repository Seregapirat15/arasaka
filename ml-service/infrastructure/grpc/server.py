import grpc
from concurrent import futures
import logging
import asyncio
import sys
import os
from typing import Dict, Any

if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

# Add shared directory to path for proto imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'shared', 'proto'))

import arasaka_pb2
import arasaka_pb2_grpc

from config.config import settings
from infrastructure.di.dependencies import get_search_usecase, get_qdrant_repository
from shared.decorators import log_grpc_calls

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ArasakaServicer(arasaka_pb2_grpc.ArasakaServiceServicer):
    """
    gRPC servicer for Arasaka question-answering service
    """
    
    def __init__(self):
        self._question_usecase = None
        self._model_loading = False
        logger.info("Arasaka service initialized")
    
    @property
    def question_usecase(self):
        """
        Get question usecase with lazy initialization
        
        Returns:
            QuestionUsecase instance for processing search requests
        """
        if self._question_usecase is None:
            self._question_usecase = get_search_usecase()
        return self._question_usecase
    
    @log_grpc_calls
    def SearchAnswers(self, request, context):
        """
        Search for similar answers using gRPC
        
        Args:
            request: gRPC SearchRequest containing query and parameters
            context: gRPC context for error handling
            
        Returns:
            gRPC SearchResponse with search results
        """
        try:
            limit = request.limit if request.limit > 0 else None
            score_threshold = request.score_threshold if request.score_threshold >= 0 else None
            
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                       future = executor.submit(
                           asyncio.run,
                           self.question_usecase.search_answers(
                               query=request.query,
                               limit=limit,
                               score_threshold=score_threshold
                           )
                       )
                       response = future.result()
                else:
                    response = loop.run_until_complete(
                        self.question_usecase.search_answers(
                            query=request.query,
                            limit=limit,
                            score_threshold=score_threshold
                        )
                    )
            except RuntimeError:
                   response = asyncio.run(
                       self.question_usecase.search_answers(
                           query=request.query,
                           limit=limit,
                           score_threshold=score_threshold
                       )
                   )
            
            results = []
            for result in response:
                grpc_result = arasaka_pb2.SearchResult(
                    id=result.answer.id,
                    answer=result.answer.text,
                    answer_id=result.answer.answer_id,
                    score=result.score,
                    metadata=result.answer.metadata or {}
                )
                results.append(grpc_result)
            
            return arasaka_pb2.SearchResponse(
                results=results,
                total_found=len(results),
                query=request.query
            )
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Search failed: {str(e)}")
            return arasaka_pb2.SearchResponse()
    
    @log_grpc_calls
    def HealthCheck(self, request, context):
        """
        Health check endpoint for service monitoring
        
        This method provides health status information about the service,
        including version, model information, and Qdrant connection status.
        
        Args:
            request: gRPC HealthRequest (empty)
            context: gRPC context for error handling
            
        Returns:
            gRPC HealthResponse with service status information
        """
        try:
            qdrant_repo = get_qdrant_repository()
            collection_info = {}
            
            try:
                import asyncio
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    collection_info = {"status": "collection_info_unavailable"}
                else:
                    collection_info = loop.run_until_complete(qdrant_repo.get_collection_info())
            except Exception as collection_error:
                logger.warning(f"Could not get collection info: {collection_error}")
                collection_info = {"error": str(collection_error)}
            
            return arasaka_pb2.HealthResponse(
                status="healthy",
                version=settings.api_version,
                model_name=settings.model_name,
                qdrant_status="connected",
                collection_info=collection_info
            )
            
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return arasaka_pb2.HealthResponse(
                status="unhealthy",
                version=settings.api_version,
                model_name=settings.model_name,
                qdrant_status="disconnected",
                collection_info={"error": str(e)}
            )


def check_and_load_data():
    """Check if Qdrant collection is empty and load data if needed"""
    try:
        qdrant_repo = get_qdrant_repository()
        
        # Check collection
        try:
            collection_info = qdrant_repo.client.get_collection(qdrant_repo.collection_name)
            points_count = collection_info.points_count
            
            if points_count > 0:
                logger.info(f"Collection already has {points_count} points, skipping data load")
                return
        except Exception as e:
            logger.warning(f"Could not check collection: {e}, will try to load data anyway")
        
        logger.info("Collection is empty or doesn't exist, loading data...")
        
        # Import and run fill script
        try:
            from tools.fill_qdrant import fill_qdrant_from_csv
            
            # __file__ is /app/ml-service/infrastructure/grpc/server.py
            # project_root should be /app/ml-service
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            csv_path = os.path.join(project_root, 'data', 'Answers__202507071202.csv')
            
            if os.path.exists(csv_path):
                logger.info(f"Loading data from {csv_path}")
                asyncio.run(fill_qdrant_from_csv(csv_path, 'Text_Cleaned', 'Id'))
                logger.info("Data loaded successfully")
            else:
                logger.warning(f"CSV file not found: {csv_path}, skipping data load")
        except Exception as e:
            logger.error(f"Failed to load data: {e}", exc_info=True)
            # Don't fail startup if data loading fails
            
    except Exception as e:
        logger.error(f"Failed to check/load data: {e}", exc_info=True)
        # Don't fail startup if data loading fails


def serve():
    print("Starting Arasaka gRPC Service...")
    print(f"System resources check:")
    print(f"   - Model: {settings.model_name}")
    print(f"   - Device: {settings.model_device}")
    print(f"   - Qdrant: {settings.qdrant_host}:{settings.qdrant_port}")
    print(f"   - Collection: {settings.qdrant_collection_name}")
    
    # Check and load data if needed (in background)
    def load_data_background():
        try:
            check_and_load_data()
        except Exception as e:
            logger.error(f"Background data load failed: {e}")
    
    import threading
    data_loader_thread = threading.Thread(target=load_data_background, daemon=True)
    data_loader_thread.start()
    
    try:
        print("Creating gRPC server...")
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        servicer = ArasakaServicer()
        arasaka_pb2_grpc.add_ArasakaServiceServicer_to_server(
            servicer, server
        )
        
        listen_addr = f"{settings.api_host}:{settings.api_port}"
        server.add_insecure_port(listen_addr)
        
        print(f"Starting gRPC server on {listen_addr}")
        logger.info(f"Starting gRPC server on {listen_addr}")
        server.start()
        
        print("Arasaka gRPC Service is running!")
        print("Ready to accept connections")
        print("Pre-loading model in background...")
        
        def preload_model():
            try:
                logger.info("Pre-loading embedding model in background...")
                _ = servicer.question_usecase
                logger.info("Model pre-loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to pre-load model: {e}")
        
        model_loader_thread = threading.Thread(target=preload_model, daemon=True)
        model_loader_thread.start()
        
        print("Press Ctrl+C to stop")
        
        try:
            server.wait_for_termination()
        except KeyboardInterrupt:
            print("\nShutting down gRPC server...")
            logger.info("Shutting down gRPC server")
            server.stop(0)
            print("Server stopped successfully")
            
    except Exception as e:
        print(f"Error starting server: {e}")
        logger.error(f"Error starting server: {e}")
        raise


if __name__ == "__main__":
    logging.basicConfig(level=getattr(logging, settings.log_level))
    serve()
