# Arasaka QA Service

gRPC service for question-answering using vector similarity search.

## Архитектура

```
max-bot/          → MAX Bot (микросервис для мессенджера)
src/              → ML Service (gRPC API + модели)
docker-compose.yml → Qdrant + ML Service
```

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

```bash
# 1. Запустить ML Service (Docker)
docker-compose up -d 

# 2. Загрузить данные в Qdrant
docker-compose exec arasaka python src/tools/fill_qdrant.py

# 3. Создать .env файл с токеном бота
cp env.example .env
# Отредактируйте .env и укажите MAX_BOT_TOKEN

# 4. Установить зависимости бота
pip install -r max-bot/requirements.txt

# 5. Запустить бота
python max-bot/bot_main.py
```

**Примечание:** Бот общается с ML Service через gRPC на порту 8001.


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

Образовательный бот для мессенджера MAX:
- Отвечает на вопросы студентов и абитуриентов
- Использует gRPC для связи с ML сервисом
- Векторный семантический поиск
- Команды: `/start`, `/help`, `/info`

Подробнее: [max-bot/README.md](max-bot/README.md)
