"""
Configuration for MAX Bot
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


class BotConfig:
    """MAX Bot configuration"""
    
    def __init__(self):
        # MAX Bot API
        self.max_bot_token: str = os.getenv("MAX_BOT_TOKEN", "")
        self.max_api_url: str = os.getenv("MAX_API_URL", "https://platform-api.max.ru")
        self.max_polling_timeout: int = int(os.getenv("MAX_POLLING_TIMEOUT", "30"))
        self.max_polling_limit: int = int(os.getenv("MAX_POLLING_LIMIT", "100"))
        
        # ML Service (gRPC)
        self.ml_service_host: str = os.getenv("ML_SERVICE_HOST", "localhost")
        self.ml_service_port: int = int(os.getenv("ML_SERVICE_PORT", "8001"))
        self.ml_service_timeout: int = int(os.getenv("ML_SERVICE_TIMEOUT", "180"))  # Increased for model loading and paraphrasing
        
        # Search settings
        self.search_limit: int = 5
        self.search_threshold: float = 0.5


settings = BotConfig()

