"""
Custom exceptions for the application
"""


class ArasakaException(Exception):
    """Base exception for Arasaka service"""
    pass


class EmbeddingServiceException(ArasakaException):
    """Exception raised by embedding service"""
    pass


class QdrantRepositoryException(ArasakaException):
    """Exception raised by Qdrant repository"""
    pass




class ConfigurationException(ArasakaException):
    """Exception raised by configuration issues"""
    pass
