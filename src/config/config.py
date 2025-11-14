"""
Configuration settings for the Arasaka Linguistic Service
"""
import os
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


class Settings:
    def __init__(self):
        self.model_name: str = "ai-forever/FRIDA"
        self.model_device: str = "cpu"
        
        self.qdrant_host: str = "localhost"
        self.qdrant_port: int = 6333
        self.qdrant_collection_name: str = "arasaka_qa"
        
        self.search_limit: int = 5
        self.similarity_threshold: float = 0.3
        
        self.api_host: str = "localhost"
        self.api_port: int = 8001
        self.api_title: str = "Arasaka Linguistic Service"
        self.api_version: str = "2.0.0"
        
        self.log_level: str = "INFO"
        
        self.enable_spell_check: bool = True
        
        # MAX Bot API settings
        self.max_api_url: str = os.getenv("MAX_API_URL", "https://platform-api.max.ru")
        self.max_bot_token: Optional[str] = os.getenv("MAX_BOT_TOKEN", None)
        self.max_polling_timeout: int = int(os.getenv("MAX_POLLING_TIMEOUT", "30"))
        self.max_polling_limit: int = int(os.getenv("MAX_POLLING_LIMIT", "100"))


settings = Settings()