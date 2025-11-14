# ML Service

gRPC service for semantic question-answering using vector similarity search with Qdrant and transformer embeddings.

## Структура

```
ml-service/
├── config/           → Configuration settings
├── domain/           → Business logic
│   └── question/     → Question domain
│       ├── entities/     → Domain entities
│       ├── repositories/ → Repository interfaces
│       ├── services/     → Domain services
│       ├── usecase/      → Use cases
│       └── delivery/dto/ → DTOs
├── infrastructure/   → Technical implementation
│   ├── db/          → Qdrant database
│   ├── di/          → Dependency injection
│   ├── grpc/        → gRPC server
│   ├── ml/          → ML/embedding services
│   └── services/    → Service implementations
├── tools/           → Utility scripts
├── data/            → Data files (CSV)
└── main.py          → Entry point
```

## Технологии

- **gRPC** - API для межсервисной коммуникации
- **Qdrant** - векторная база данных
- **ai-forever/FRIDA** - модель для эмбеддингов
- **PyTorch** - ML framework

## Конфигурация

Настройки в `ml-service/config/config.py` или через `.env`:

```bash
# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333

# ML Service API
API_HOST=0.0.0.0
API_PORT=8001

# Model
MODEL_NAME=ai-forever/FRIDA
```

## Запуск

### В Docker (рекомендуется)

```bash
# Запустить Qdrant + ML Service
docker-compose up -d

# Загрузить данные
docker-compose exec ml-service python ml-service/tools/fill_qdrant.py

# Проверить статус
docker-compose logs ml-service
```

### Локально

```bash
# Установить зависимости
pip install -r ml-service/requirements.txt

# Запустить Qdrant
docker-compose up -d qdrant

# Загрузить данные
python ml-service/tools/fill_qdrant.py

# Запустить сервис
python ml-service/main.py
```

## gRPC API

### SearchAnswers

Поиск похожих ответов по запросу:

```python
request = SearchRequest(
    query="как поступить в университет?",
    limit=5,
    score_threshold=0.3
)
response = stub.SearchAnswers(request)
```

### HealthCheck

Проверка состояния сервиса:

```python
request = HealthRequest()
response = stub.HealthCheck(request)
```

## Proto файлы

gRPC proto определения находятся в `shared/proto/arasaka.proto`.

Для генерации Python файлов:

```bash
cd shared/proto
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. arasaka.proto
```

## Данные

Данные загружаются из CSV файлов в `ml-service/data/`:
- `Questions__*.csv` - вопросы
- `Answers__*.csv` - ответы

Используйте скрипт `ml-service/tools/fill_qdrant.py` для загрузки в Qdrant.

## Разработка

### Архитектура

Сервис использует Clean Architecture:
- **Domain** - бизнес-логика, независимая от фреймворков
- **Infrastructure** - технические детали (БД, API, ML)
- **DI Container** - внедрение зависимостей

### Добавление новых функций

1. Определите интерфейс в `domain/`
2. Реализуйте в `infrastructure/`
3. Зарегистрируйте в `infrastructure/di/dependencies.py`
4. Обновите proto файл при необходимости

