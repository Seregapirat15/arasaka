"""
Configuration settings for the Arasaka Linguistic Service
"""
import os
from typing import Optional


class Settings:
    def __init__(self):
        self.model_name: str = os.getenv("MODEL_NAME", "ai-forever/FRIDA")
        self.model_device: str = "cpu"
        
        self.qdrant_host: str = os.getenv("QDRANT_HOST", "localhost")
        self.qdrant_port: int = int(os.getenv("QDRANT_PORT", "6333"))
        self.qdrant_collection_name: str = "arasaka_qa"
        
        self.search_limit: int = 5
        self.similarity_threshold: float = 0.3
        
        self.api_host: str = os.getenv("API_HOST", "0.0.0.0")
        self.api_port: int = int(os.getenv("API_PORT", "8001"))
        self.api_title: str = "Arasaka Linguistic Service"
        self.api_version: str = "2.0.0"
        
        self.log_level: str = "INFO"
        
        self.enable_spell_check: bool = True


settings = Settings()