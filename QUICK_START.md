# 🚀 Quick Start Guide

Самая быстрая инструкция для запуска проекта Arasaka.

## ⏱️ 5-минутный старт

```bash
# 1. Клонируйте и перейдите в проект
git clone <repo-url>
cd arasaka

# 2. Создайте .env
cp env.example .env
# Отредактируйте .env и добавьте MAX_BOT_TOKEN

# 3. Запустите ML Service + Qdrant
docker-compose up -d

# 4. Дождитесь загрузки модели (5-10 мин при первом запуске)
docker-compose logs -f ml-service

# 5. Загрузите данные
docker-compose exec ml-service python ml-service/tools/fill_qdrant.py

# 6. Установите зависимости
pip install -r requirements.txt

# 7. Запустите бота
python max-bot/bot_main.py
```