from typing import Optional
from logging import getLogger
from infrastructure.db.qdrant import QdrantRepository
from infrastructure.ml.embedding_service_impl import EmbeddingServiceImpl
from infrastructure.services.question_service_impl import QuestionServiceImpl
from domain.question.usecase.question_usecase import QuestionUsecase

logger = getLogger(__name__)


class DIContainer:
    def __init__(self):
        self._qdrant_repository: Optional[QdrantRepository] = None
        self._embedding_service: Optional[EmbeddingServiceImpl] = None
        self._paraphrase_service: Optional[object] = None
        self._question_service: Optional[QuestionServiceImpl] = None
        self._search_usecase: Optional[QuestionUsecase] = None
    
    def get_qdrant_repository(self) -> QdrantRepository:
        if self._qdrant_repository is None:
            self._qdrant_repository = QdrantRepository()
        return self._qdrant_repository
    
    def get_embedding_service(self) -> EmbeddingServiceImpl:
        if self._embedding_service is None:
            self._embedding_service = EmbeddingServiceImpl()
        return self._embedding_service
    
    def get_paraphrase_service(self) -> Optional[object]:
        """
        Get paraphrase service, trying to load T5 model first, falling back to simple service
        """
        if self._paraphrase_service is None:
            try:
                from infrastructure.paraphrasing.paraphrase_service import ParaphraseService
                from config.config import settings
                
                # Try to load T5 model
                device = getattr(settings, 'model_device', 'cpu')
                self._paraphrase_service = ParaphraseService(
                    model_name="cointegrated/rut5-base-paraphraser",
                    device=device,
                    similarity_threshold=0.7
                )
                logger.info("ParaphraseService (T5) loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load ParaphraseService (T5): {e}, using SimpleParaphraseService")
                try:
                    from infrastructure.paraphrasing.paraphrase_service import SimpleParaphraseService
                    self._paraphrase_service = SimpleParaphraseService()
                    logger.info("SimpleParaphraseService loaded as fallback")
                except Exception as e2:
                    logger.error(f"Failed to load SimpleParaphraseService: {e2}")
                    self._paraphrase_service = None
        
        return self._paraphrase_service
    
    def get_question_service(self) -> QuestionServiceImpl:
        if self._question_service is None:
            from config.config import settings
            paraphrase_service = None
            use_paraphrasing = False
            
            # Only enable paraphrasing if explicitly enabled in config
            if settings.use_paraphrasing:
                paraphrase_service = self.get_paraphrase_service()
                use_paraphrasing = paraphrase_service is not None
            
            self._question_service = QuestionServiceImpl(
                self.get_embedding_service(),
                self.get_qdrant_repository(),
                paraphrase_service=paraphrase_service,
                use_paraphrasing=use_paraphrasing
            )
        return self._question_service
    
    
    def get_search_usecase(self) -> QuestionUsecase:
        if self._search_usecase is None:
            self._search_usecase = QuestionUsecase(self.get_question_service())
        return self._search_usecase


_container = DIContainer()


def get_qdrant_repository():
    return _container.get_qdrant_repository()


def get_embedding_service():
    return _container.get_embedding_service()




def get_question_service():
    return _container.get_question_service()


def get_search_usecase():
    return _container.get_search_usecase()


def get_paraphrase_service():
    return _container.get_paraphrase_service()
