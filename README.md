# Arasaka QA Service

gRPC service for question-answering using vector similarity search + MAX Bot integration.

## Quick Start

### Запуск QA сервиса (gRPC)

```bash
# Start services
docker-compose up -d

# Load data
python src/tools/fill_qdrant.py

# Test
python test_search.py
```

### Запуск бота MAX

**Ручной способ:**
```bash
# 1. Запустить Qdrant
docker-compose up -d 

# 2. Загрузить данные (если еще не загружены)
python src/tools/fill_qdrant.py

# 3. Создать .env файл с токеном бота
cp env.example .env
# Отредактируйте .env и укажите MAX_BOT_TOKEN

# 4. Установить зависимости
pip install -r src/requirements.txt
pip install -r src/infrastructure/max/requirements.txt

# 5. Запустить бота
python src/infrastructure/max/bot_main.py
```

**Примечание:** Бот использует usecase напрямую, gRPC сервис не обязателен для работы бота.


## Configuration

Edit `src/config/config.py` or `.env` file:
- API: `localhost:8001`
- Qdrant: `localhost:6333`
- Model: `ai-forever/FRIDA`
- Collection name: `arasaka_qa`
- MAX Bot Token: `MAX_BOT_TOKEN` (в .env)

## API

gRPC service on port 8001:

- `SearchAnswers(query, limit, score_threshold)` - search for similar answers
- `HealthCheck()` - service health status

## MAX Bot

Бот для мессенджера MAX:
- Отвечает на вопросы студентов
- Использует семантический поиск
- Команды: `/start`, `/help`, `/info`

Подробнее: [src/infrastructure/max/README.md](src/infrastructure/max/README.md)
