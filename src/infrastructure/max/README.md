# Arasaka — MAX Bot

Образовательный помощник в мессенджере MAX для студентов и абитуриентов. Отвечает на вопросы об обучении, используя AI-поиск по базе знаний.

## Установка зависимостей

```bash
pip install -r src/infrastructure/max/requirements.txt
```

## Настройка

Создайте файл `.env` в корне проекта (скопируйте из `env.example`):

```bash
cp env.example .env
```

Отредактируйте `.env` и укажите токен бота:

```env
MAX_BOT_TOKEN=your_token_here
MAX_API_URL=https://platform-api.max.ru
MAX_POLLING_TIMEOUT=30
MAX_POLLING_LIMIT=100
```

## Запуск

```bash
python src/infrastructure/max/bot_main.py
```

**Технология:**
- AI-поиск по базе образовательных документов
- Векторный поиск для понимания смысла вопросов
- Работа с естественным языком

**Команды:**
- `/start` - Начать работу с ботом
- `/help` - Примеры вопросов
- `/info` - О боте


## Конфигурация

Настройки находятся в `.env`:

- `MAX_BOT_TOKEN` - Токен бота (обязательно)
- `MAX_API_URL` - URL API (по умолчанию: https://platform-api.max.ru)
- `MAX_POLLING_TIMEOUT` - Таймаут для Long Polling (по умолчанию: 30 секунд)
- `MAX_POLLING_LIMIT` - Максимальное количество обновлений за запрос (по умолчанию: 100)


