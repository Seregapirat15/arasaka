"""
Main entry point for MAX bot using umaxbot
"""
import logging
import asyncio
import sys

from maxbot.bot import Bot
from maxbot.dispatcher import Dispatcher

from config import settings
from handlers import setup_handlers

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Отключаем DEBUG логи от сторонних библиотек
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('asyncio').setLevel(logging.WARNING)
logging.getLogger('maxbot').setLevel(logging.INFO)


def main():
    """
    Start MAX bot using umaxbot
    """
    logger.info("Starting Arasaka MAX Bot...")
    
    # Check if token is set
    if not settings.max_bot_token:
        logger.error("MAX_BOT_TOKEN environment variable is not set!")
        logger.error("Please set it before running the bot:")
        logger.error("  export MAX_BOT_TOKEN=your_token_here")
        logger.error("  Or create .env file with MAX_BOT_TOKEN=your_token")
        sys.exit(1)
    
    try:
        # Create bot and dispatcher
        bot = Bot(settings.max_bot_token)
        dp = Dispatcher(bot)
        
        # Setup handlers
        setup_handlers(dp, bot)
        
        # Verify bot token (umaxbot does this automatically on start_polling)
        logger.info("Bot and dispatcher initialized successfully")
        
        logger.info("Starting Long Polling...")
        logger.info("Press Ctrl+C to stop")
        
        # Start polling (async function)
        asyncio.run(dp.run_polling())
        
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Failed to start bot: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()




