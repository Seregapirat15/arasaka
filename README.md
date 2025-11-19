#  Quick Start Guide

Быстрая инструкция для запуска проекта Arasaka.

## 1. Клонирование проекта

```bash
git clone <repo-url>
cd arasaka
```

## 2. Настройка окружения

```bash
cp env.example .env
```

Создайте `.env` и добавьте `MAX_BOT_TOKEN`.

## 3. Запуск ML Service и Qdrant

```bash
docker-compose up -d
```

## 4. Ожидание загрузки модели

При первом запуске модель загружается 5-10 минут:

```bash
docker-compose logs -f ml-service
```

## 5. Загрузка данных

```bash
python ml-service/tools/fill_qdrant.py
```

## 6. Установка зависимостей

```bash
pip install -r requirements.txt
```

## 7. Запуск бота

```bash
python max-bot/bot_main.py
```
