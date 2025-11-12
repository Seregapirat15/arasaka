# Arasaka QA Service

gRPC service for question-answering using vector similarity search.

## Quick Start

```bash
# Start services
docker-compose up -d

# Load data
python src/tools/fill_qdrant.py

# Test
python test_search.py
```

## Configuration

Edit `src/config/config.py`:
- API: `localhost:8001`
- Qdrant: `localhost:6333`
- Model: `ai-forever/FRIDA`
- Collection name: `arasaka_qa`

## API

gRPC service on port 8001:

- `SearchAnswers(query, limit, score_threshold)` - search for similar answers
- `HealthCheck()` - service health status
