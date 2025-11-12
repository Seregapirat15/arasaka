import sys
import asyncio
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
experiment_src = Path(__file__).parent / "src"
sys.path.insert(0, str(experiment_src))

from infrastructure.db.qdrant import QdrantRepository
from infrastructure.ml.embedding_service_impl import EmbeddingServiceImpl
from paraphrase_service import SimpleParaphraseService
from voting_ranker import VotingRanker, AnswerResult


async def quick_test():
    print("=" * 60)
    print("Быстрый тест системы парафразирования")
    print("=" * 60)
    
    print("\n1. Проверка подключения к Qdrant...")
    try:
        qdrant_repo = QdrantRepository()
        print(f"   [OK] Qdrant подключен: {qdrant_repo.host}:{qdrant_repo.port}")
    except Exception as e:
        print(f"   [ERROR] Ошибка подключения к Qdrant: {e}")
        print("   Убедитесь, что Qdrant запущен (docker-compose up -d)")
        return
    
    print("\n2. Проверка embedding service...")
    try:
        embedding_service = EmbeddingServiceImpl()
        test_embedding = embedding_service.encode_text("тестовый вопрос")
        print(f"   [OK] Embedding service работает (размерность: {len(test_embedding)})")
    except Exception as e:
        print(f"   [ERROR] Ошибка embedding service: {e}")
        return
    
    print("\n3. Проверка генерации парафразов...")
    try:
        paraphrase_service = SimpleParaphraseService()
        test_query = "Как работает система?"
        paraphrases = paraphrase_service.generate_paraphrases(test_query, num_paraphrases=3)
        print(f"   [OK] Парафразы сгенерированы для: '{test_query}'")
        for i, para in enumerate(paraphrases, 1):
            print(f"      {i}. {para}")
    except Exception as e:
        print(f"   [ERROR] Ошибка генерации парафразов: {e}")
        return
    
    print("\n4. Проверка базового поиска...")
    try:
        query = "как зарегистрироваться"
        query_embedding = embedding_service.encode_text(query)
        results = await qdrant_repo.find_similar_answers(
            query_embedding=query_embedding,
            limit=3,
            score_threshold=0.3
        )
        print(f"   [OK] Найдено результатов: {len(results)}")
        if results:
            print(f"      Пример: {results[0].text[:100]}...")
        else:
            print("      [WARNING] Результатов не найдено. Убедитесь, что данные загружены в Qdrant")
    except Exception as e:
        print(f"   [ERROR] Ошибка поиска: {e}")
        return
    
    print("\n5. Проверка голосующей схемы...")
    try:
        ranker = VotingRanker(voting_method="weighted")
        test_results = {
            "парафраз 1": [
                AnswerResult("ans1", "ответ 1", 0.8),
                AnswerResult("ans2", "ответ 2", 0.7)
            ],
            "парафраз 2": [
                AnswerResult("ans1", "ответ 1", 0.75),
                AnswerResult("ans3", "ответ 3", 0.6)
            ]
        }
        ranked = ranker.rank_answers(test_results)
        print(f"   [OK] Голосующая схема работает (ранжировано: {len(ranked)} ответов)")
    except Exception as e:
        print(f"   [ERROR] Ошибка голосующей схемы: {e}")
        return
    
    print("\n" + "=" * 60)
    print("[SUCCESS] Все компоненты работают корректно!")
    print("=" * 60)
    print("\nСледующие шаги:")
    print("1. Подготовьте тестовые данные в data/test_queries.csv")
    print("2. Запустите полный эксперимент: python src/experiment_runner.py")


if __name__ == "__main__":
    asyncio.run(quick_test())
