from typing import Optional
from infrastructure.db.qdrant import QdrantRepository
from infrastructure.ml.embedding_service_impl import EmbeddingServiceImpl
from infrastructure.services.question_service_impl import QuestionServiceImpl
from domain.question.usecase.question_usecase import QuestionUsecase


class DIContainer:
    def __init__(self):
        self._qdrant_repository: Optional[QdrantRepository] = None
        self._embedding_service: Optional[EmbeddingServiceImpl] = None
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
    
    def get_question_service(self) -> QuestionServiceImpl:
        if self._question_service is None:
            self._question_service = QuestionServiceImpl(
                self.get_embedding_service(),
                self.get_qdrant_repository()
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
