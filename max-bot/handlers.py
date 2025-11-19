"""
Handlers for MAX bot using umaxbot
"""
import logging

from maxbot.dispatcher import Dispatcher
from maxbot.bot import Bot
from maxbot.types import Message, InlineKeyboardMarkup, InlineKeyboardButton

from config import settings

logger = logging.getLogger(__name__)

# Опциональный импорт ML сервиса через gRPC
try:
    from grpc_client import get_ml_client
    
    # Check ML service availability
    ml_client = get_ml_client()
    ML_SERVICE_AVAILABLE = ml_client.health_check()
    
    if ML_SERVICE_AVAILABLE:
        logger.info("ML service is available via gRPC")
    else:
        logger.warning("ML service health check failed")
except Exception as e:
    logger.warning(f"ML service not available: {e}. Bot will work but won't answer questions.")
    ML_SERVICE_AVAILABLE = False
    ml_client = None


def setup_handlers(dp: Dispatcher, bot: Bot):
    """
    Setup all bot handlers
    
    Args:
        dp: Dispatcher instance
        bot: Bot instance
    """
    
    @dp.message()
    async def handle_message(message: Message):
        """Handle all other messages (questions)"""
        text = message.text.strip() if message.text else ""
        chat_id = message.chat.id
        
        if not text:
            return
        
        # Handle commands
        if text == '/start':
            logger.info(f"Received /start from chat {chat_id}")
            await bot.send_message(
                chat_id=chat_id,
                text=(
                    "👋 Привет! Я — Arasaka, помощник по образовательным вопросам.\n\n"
                    "Помогу найти ответы на вопросы об учёбе:\n"
                    "📚 Программы обучения и специальности\n"
                    "📝 Экзамены и требования к поступлению\n"
                    "📅 Сроки подачи документов\n"
                    "🏠 Общежития и стипендии\n"
                    "📖 Учебные планы и расписание\n\n"
                    "Просто задайте свой вопрос!\n\n"
                    "💡 Команды:\n"
                    "/help - примеры вопросов\n"
                    "/info - о боте"
                ),
                notify=True
            )
            return
        
        if text == '/help':
            logger.info(f"Received /help from chat {chat_id}")
            await bot.send_message(
                chat_id=chat_id,
                text=(
                    "📚 Примеры вопросов:\n\n"
                    "О поступлении:\n"
                    "• \"Какие документы нужны для поступления?\"\n"
                    "• \"Когда начинается приём документов?\"\n"
                    "• \"Какие экзамены нужно сдавать на программирование?\"\n"
                    "• \"Есть ли бюджетные места?\"\n\n"
                    "Об обучении:\n"
                    "• \"Какие специальности есть в вузе?\"\n"
                    "• \"Сколько длится обучение?\"\n"
                    "• \"Есть ли общежитие для иногородних?\"\n"
                    "• \"Какая стипендия для отличников?\"\n\n"
                    "💡 Формулируйте вопрос конкретно для лучшего результата!"
                ),
                notify=True
            )
            return
        
        if text == '/info':
            logger.info(f"Received /info from chat {chat_id}")
            await bot.send_message(
                chat_id=chat_id,
                text=(
                    "ℹ️ О боте Arasaka:\n\n"
                    "Я — образовательный помощник на базе искусственного интеллекта. "
                    "Помогаю студентам и абитуриентам быстро находить нужную информацию.\n\n"
                    "🎓 Что я знаю:\n"
                    "• Правила приёма и поступления\n"
                    "• Образовательные программы и специальности\n"
                    "• Требования к документам и экзаменам\n"
                    "• Информацию об общежитиях и стипендиях\n"
                    "• Учебные планы и расписания\n\n"
                    "📚 Все ответы основаны на официальной информации учебного заведения"
                ),
                notify=True
            )
            return
        
        # All other messages are questions
        logger.info(f"Received question from chat {chat_id}: {text[:50]}...")
        
        # Send "thinking" message
        thinking_msg = await bot.send_message(
            chat_id=chat_id,
            text="🔍 Ищу ответ...",
            notify=False
        )
        
        thinking_msg_id = thinking_msg.get("message_id") if isinstance(thinking_msg, dict) else None
        
        # Check if ML service is available
        if not ML_SERVICE_AVAILABLE:
            error_text = (
                "⚠️ Сервис поиска ответов временно недоступен.\n\n"
                "Пожалуйста, попробуйте позже или используйте команды /start, /help, /info"
            )
            
            if thinking_msg_id:
                try:
                    await bot.edit_message(
                        chat_id=chat_id,
                        message_id=thinking_msg_id,
                        text=error_text
                    )
                except:
                    await bot.send_message(
                        chat_id=chat_id,
                        text=error_text,
                        notify=True
                    )
            else:
                await bot.send_message(
                    chat_id=chat_id,
                    text=error_text,
                    notify=True
                )
            return
        
        try:
            # Search for answers via gRPC ML service
            results = ml_client.search_answers(
                query=text,
                limit=settings.search_limit,
                score_threshold=settings.search_threshold
            )
            
            logger.info(f"Search query: '{text}', found {len(results) if results else 0} results")
            
            if results and len(results) > 0:
                # Get the best answer
                best_result = results[0]
                answer_text = best_result['answer']['text']
                
                # Format response
                response = f"💡 Ответ:\n\n{answer_text}"
                
                # Update or send message with answer
                if thinking_msg_id:
                    try:
                        await bot.edit_message(
                            chat_id=chat_id,
                            message_id=thinking_msg_id,
                            text=response,
                            format="markdown"
                        )
                    except:
                        # If edit fails, send new message
                        await bot.send_message(
                            chat_id=chat_id,
                            text=response,
                            notify=True,
                            format="markdown"
                        )
                else:
                    await bot.send_message(
                        chat_id=chat_id,
                        text=response,
                        notify=True,
                        format="markdown"
                    )
                
                logger.info(f"Sent answer to chat {chat_id}")
            else:
                # No results found
                error_text = (
                    "❌ К сожалению, я не смог найти подходящий ответ на ваш вопрос.\n\n"
                    "Попробуйте переформулировать вопрос или используйте другие ключевые слова."
                )
                
                if thinking_msg_id:
                    try:
                        await bot.edit_message(
                            chat_id=chat_id,
                            message_id=thinking_msg_id,
                            text=error_text
                        )
                    except:
                        await bot.send_message(
                            chat_id=chat_id,
                            text=error_text,
                            notify=True
                        )
                else:
                    await bot.send_message(
                        chat_id=chat_id,
                        text=error_text,
                        notify=True
                    )
                
                logger.warning(f"No results found for query '{text}' from chat {chat_id}")
        
        except Exception as e:
            logger.error(f"Error processing question: {e}", exc_info=True)
            
            # Send error message
            error_text = (
                "⚠️ Произошла ошибка при поиске ответа.\n\n"
                "Попробуйте позже или обратитесь к администратору."
            )
            
            if thinking_msg_id:
                try:
                    await bot.edit_message(
                        chat_id=chat_id,
                        message_id=thinking_msg_id,
                        text=error_text
                    )
                except:
                    await bot.send_message(
                        chat_id=chat_id,
                        text=error_text,
                        notify=True
                    )
            else:
                await bot.send_message(
                    chat_id=chat_id,
                    text=error_text,
                    notify=True
                )
