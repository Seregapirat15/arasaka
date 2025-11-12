import sys
import json
import asyncio
import time
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from logging import getLogger, basicConfig, INFO
from dataclasses import dataclass, asdict
from collections import defaultdict

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from infrastructure.db.qdrant import QdrantRepository
from infrastructure.ml.embedding_service_impl import EmbeddingServiceImpl

experiment_src = Path(__file__).parent
sys.path.insert(0, str(experiment_src))
from paraphrase_service import ParaphraseService, SimpleParaphraseService
from voting_ranker import VotingRanker, AnswerResult

basicConfig(level=INFO)
logger = getLogger(__name__)


@dataclass
class ExperimentResult:
    query: str
    method: str
    results: List[Dict]
    execution_time: float
    num_paraphrases: Optional[int] = None
    voting_method: Optional[str] = None


@dataclass
class Metrics:
    precision_at_1: float
    precision_at_3: float
    precision_at_5: float
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr: float
    ndcg_at_5: float


class ExperimentRunner:
    
    def __init__(
        self,
        paraphrase_service: Optional[ParaphraseService] = None,
        use_simple_paraphrase: bool = False
    ):
        self.embedding_service = EmbeddingServiceImpl()
        self.qdrant_repo = QdrantRepository()
        
        if use_simple_paraphrase:
            self.paraphrase_service = SimpleParaphraseService()
        elif paraphrase_service:
            self.paraphrase_service = paraphrase_service
        else:
            self.paraphrase_service = SimpleParaphraseService()
    
    async def baseline_search(self, query: str, limit: int = 5) -> List[Dict]:
        try:
            query_embedding = self.embedding_service.encode_text(query)
            answers = await self.qdrant_repo.find_similar_answers(
                query_embedding=query_embedding,
                limit=limit,
                score_threshold=0.3
            )
            
            results = []
            for answer in answers:
                score = getattr(answer, 'score', 0.0)
                results.append({
                    "answer_id": answer.answer_id,
                    "answer_text": answer.text,
                    "score": float(score)
                })
            
            return results
        except Exception as e:
            logger.error(f"Baseline search failed: {e}")
            return []
    
    async def paraphrasing_search(
        self,
        query: str,
        num_paraphrases: int = 3,
        voting_method: str = "weighted",
        limit_per_paraphrase: int = 5
    ) -> List[Dict]:
        try:
            paraphrases = self.paraphrase_service.generate_paraphrases(
                query,
                num_paraphrases=num_paraphrases
            )
            
            if not paraphrases:
                logger.warning(f"No paraphrases generated for query: {query}")
                return await self.baseline_search(query, limit=limit_per_paraphrase)
            
            if len(paraphrases) == 1 and paraphrases[0].lower() == query.lower():
                logger.warning(f"Only original query returned as paraphrase, using baseline")
                return await self.baseline_search(query, limit=limit_per_paraphrase)
            
            search_results_by_paraphrase = {}
            
            for paraphrase in paraphrases:
                query_embedding = self.embedding_service.encode_text(paraphrase)
                answers = await self.qdrant_repo.find_similar_answers(
                    query_embedding=query_embedding,
                    limit=limit_per_paraphrase,
                    score_threshold=0.2
                )
                
                answer_results = []
                for answer in answers:
                    score = getattr(answer, 'score', 0.5)
                    answer_results.append(AnswerResult(
                        answer_id=answer.answer_id,
                        answer_text=answer.text,
                        score=float(score)
                    ))
                
                search_results_by_paraphrase[paraphrase] = answer_results
            
            ranker = VotingRanker(voting_method=voting_method)
            ranked_results = ranker.rank_answers(search_results_by_paraphrase)
            
            results = []
            for answer_id, answer_text, avg_score, max_score, ranking_score in ranked_results:
                results.append({
                    "answer_id": answer_id,
                    "answer_text": answer_text,
                    "avg_score": avg_score,
                    "max_score": max_score,
                    "ranking_score": ranking_score
                })
            
            return results[:limit_per_paraphrase]
            
        except Exception as e:
            logger.error(f"Paraphrasing search failed: {e}")
            return []
    
    async def run_experiment(
        self,
        query: str,
        ground_truth: Optional[List[str]] = None
    ) -> Dict[str, ExperimentResult]:
        results = {}
        
        start_time = time.time()
        baseline_results = await self.baseline_search(query, limit=5)
        baseline_time = time.time() - start_time
        
        results["baseline"] = ExperimentResult(
            query=query,
            method="baseline",
            results=baseline_results,
            execution_time=baseline_time
        )
        
        start_time = time.time()
        paraphrasing_results = await self.paraphrasing_search(
            query,
            num_paraphrases=5,
            voting_method="weighted",
            limit_per_paraphrase=5
        )
        paraphrasing_time = time.time() - start_time
        
        baseline_ids = {r["answer_id"] for r in baseline_results}
        paraphrasing_ids = {r["answer_id"] for r in paraphrasing_results}
        
        common_ids = baseline_ids & paraphrasing_ids
        only_baseline = baseline_ids - paraphrasing_ids
        only_paraphrasing = paraphrasing_ids - baseline_ids
        
        self._print_comparison(
            query,
            baseline_results,
            paraphrasing_results,
            baseline_time,
            paraphrasing_time,
            common_ids,
            only_baseline,
            only_paraphrasing
        )
        
        results["paraphrasing"] = ExperimentResult(
            query=query,
            method="paraphrasing",
            results=paraphrasing_results,
            execution_time=paraphrasing_time,
            num_paraphrases=5,
            voting_method="weighted"
        )
        
        return results
    
    def _print_comparison(
        self,
        query: str,
        baseline_results: List[Dict],
        paraphrasing_results: List[Dict],
        baseline_time: float,
        paraphrasing_time: float,
        common_ids: set,
        only_baseline: set,
        only_paraphrasing: set
    ):
        print("\n" + "=" * 80)
        print(f"ВОПРОС: {query}")
        print("=" * 80)
        
        print(f"\nВРЕМЯ ВЫПОЛНЕНИЯ:")
        print(f"  Baseline:    {baseline_time:.2f} сек")
        print(f"  Paraphrasing: {paraphrasing_time:.2f} сек ({paraphrasing_time/baseline_time:.1f}x медленнее)")
        
        print(f"\nСТАТИСТИКА:")
        print(f"  Baseline нашел:    {len(baseline_results)} ответов")
        print(f"  Paraphrasing нашел: {len(paraphrasing_results)} ответов")
        print(f"  Общих ответов:      {len(common_ids)}")
        print(f"  Только в baseline:  {len(only_baseline)}")
        print(f"  Только в paraphrasing: {len(only_paraphrasing)} {'[НОВЫЕ]' if len(only_paraphrasing) > 0 else ''}")
        
        if len(only_paraphrasing) > 0:
            print(f"\nНОВЫЕ ОТВЕТЫ ОТ PARAPHRASING: {sorted(only_paraphrasing)}")
        
        print(f"\nТОП-5 BASELINE:")
        for i, result in enumerate(baseline_results[:5], 1):
            marker = "[ОБЩИЙ]" if result["answer_id"] in common_ids else "[ТОЛЬКО BASELINE]"
            print(f"  {i}. {marker} ID: {result['answer_id']:>3} | Score: {result['score']:.3f} | {result['answer_text'][:60]}...")
        
        print(f"\nТОП-5 PARAPHRASING:")
        for i, result in enumerate(paraphrasing_results[:5], 1):
            if result["answer_id"] in only_paraphrasing:
                marker = "[НОВЫЙ]"
            elif result["answer_id"] in common_ids:
                marker = "[ОБЩИЙ]"
            else:
                marker = "[ДРУГОЙ]"
            avg_score = result.get("avg_score", result.get("score", 0))
            max_score = result.get("max_score", avg_score)
            ranking_score = result.get("ranking_score", 0)
            print(f"  {i:>2}. {marker} ID: {result['answer_id']:>3} | Avg: {avg_score:.3f} | Max: {max_score:.3f} | Rank: {ranking_score:.3f} | {result['answer_text'][:60]}...")
        
        print("\n" + "=" * 80)
    
    def calculate_metrics(
        self,
        results: List[Dict],
        ground_truth: List[str],
        k: int = 5
    ) -> Metrics:
        if not ground_truth:
            return Metrics(0, 0, 0, 0, 0, 0, 0, 0)
        
        ground_truth_set = set(ground_truth)
        result_ids = [r["answer_id"] for r in results[:k]]
        
        relevant_at_k = sum(1 for aid in result_ids if aid in ground_truth_set)
        precision_at_k = relevant_at_k / len(result_ids) if result_ids else 0
        
        recall_at_k = relevant_at_k / len(ground_truth_set) if ground_truth_set else 0
        
        mrr = 0.0
        for i, result in enumerate(results):
            if result["answer_id"] in ground_truth_set:
                mrr = 1.0 / (i + 1)
                break
        
        dcg = 0.0
        for i, result in enumerate(results[:k]):
            if result["answer_id"] in ground_truth_set:
                dcg += 1.0 / (i + 1)
        idcg = sum(1.0 / (i + 1) for i in range(min(len(ground_truth_set), k)))
        ndcg_at_k = dcg / idcg if idcg > 0 else 0.0
        
        return Metrics(
            precision_at_1=precision_at_k if k >= 1 else 0,
            precision_at_3=precision_at_k if k >= 3 else 0,
            precision_at_5=precision_at_k if k >= 5 else 0,
            recall_at_1=recall_at_k if k >= 1 else 0,
            recall_at_3=recall_at_k if k >= 3 else 0,
            recall_at_5=recall_at_k if k >= 5 else 0,
            mrr=mrr,
            ndcg_at_5=ndcg_at_k
        )


async def main():
    try:
        paraphrase_service = ParaphraseService(
            model_name="cointegrated/rut5-base-paraphraser",
            device="cpu",
            similarity_threshold=0.4
        )
        runner = ExperimentRunner(paraphrase_service=paraphrase_service)
    except Exception as e:
        logger.warning(f"Failed to load ParaphraseService, using SimpleParaphraseService: {e}")
        runner = ExperimentRunner(use_simple_paraphrase=True)
    
    test_queries = [
        "Несёт ли обучение на ЦК какие-то обязательства, помимо присутствия на занятиях и выполнения работ? Например, необходимость отработать какое-то время в какой-то организации.",
        "Можно ли будет стажировку ЦК зачесть в качестве обязательной производственной практики, на которую вуз направляет студентов после 3-го курса обучения?",
        "Получится ли совместить написание дипломной работы по окончании обучения с прохождением военных сборов?",
        "Расскажите подробнее, чему научат на программах DevOps-инженер, Digital-маркетолог, Руководитель продукта?",
        "Есть ли расписание занятий? Планируется ли оповещение студентов о сроках, которые даются им на прохождение модуля? Есть ли сроки проведения всех мероприятий, связанных с обучением?",
        "В личном кабинете иннополиса отсутствует моя программа, из-за чего не могу пройти ассесмент.",
        "Как понять, как и в каком формате можно будет приступить к прохождению курса?",
        "Как проходит входное тестирование? Нужно ли к нему как-то готовиться?",
    ]
    
    all_results = []
    total_stats = {
        "baseline_time": 0,
        "paraphrasing_time": 0,
        "baseline_total_answers": 0,
        "paraphrasing_total_answers": 0,
        "total_common": 0,
        "total_new_from_paraphrasing": 0,
        "queries_with_new_answers": 0
    }
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'#' * 80}")
        print(f"ЭКСПЕРИМЕНТ {i}/{len(test_queries)}")
        print(f"{'#' * 80}")
        
        results = await runner.run_experiment(query)
        all_results.append(results)
        
        baseline_ids = {r["answer_id"] for r in results["baseline"].results}
        paraphrasing_ids = {r["answer_id"] for r in results["paraphrasing"].results}
        common_ids = baseline_ids & paraphrasing_ids
        only_paraphrasing = paraphrasing_ids - baseline_ids
        
        total_stats["baseline_time"] += results["baseline"].execution_time
        total_stats["paraphrasing_time"] += results["paraphrasing"].execution_time
        total_stats["baseline_total_answers"] += len(baseline_ids)
        total_stats["paraphrasing_total_answers"] += len(paraphrasing_ids)
        total_stats["total_common"] += len(common_ids)
        total_stats["total_new_from_paraphrasing"] += len(only_paraphrasing)
        if len(only_paraphrasing) > 0:
            total_stats["queries_with_new_answers"] += 1
    
    print("\n" + "=" * 80)
    print("ИТОГОВАЯ СТАТИСТИКА ПО ВСЕМ ЭКСПЕРИМЕНТАМ")
    print("=" * 80)
    print(f"\nВРЕМЯ ВЫПОЛНЕНИЯ:")
    print(f"  Baseline:    {total_stats['baseline_time']:.2f} сек (среднее: {total_stats['baseline_time']/len(test_queries):.2f} сек/запрос)")
    print(f"  Paraphrasing: {total_stats['paraphrasing_time']:.2f} сек (среднее: {total_stats['paraphrasing_time']/len(test_queries):.2f} сек/запрос)")
    print(f"  Замедление:   {total_stats['paraphrasing_time']/total_stats['baseline_time']:.1f}x")
    
    print(f"\nНАЙДЕННЫЕ ОТВЕТЫ:")
    print(f"  Baseline:    {total_stats['baseline_total_answers']} уникальных ответов")
    print(f"  Paraphrasing: {total_stats['paraphrasing_total_answers']} уникальных ответов")
    print(f"  Общих:       {total_stats['total_common']} ответов")
    print(f"  Новых от Paraphrasing: {total_stats['total_new_from_paraphrasing']} ответов")
    
    print(f"\nУЛУЧШЕНИЯ:")
    print(f"  Запросов с новыми ответами: {total_stats['queries_with_new_answers']}/{len(test_queries)} ({100*total_stats['queries_with_new_answers']/len(test_queries):.0f}%)")
    print(f"  Среднее новых ответов на запрос: {total_stats['total_new_from_paraphrasing']/len(test_queries):.1f}")
    
    if total_stats['total_new_from_paraphrasing'] > 0:
        print(f"\nВЫВОД: Paraphrasing нашел {total_stats['total_new_from_paraphrasing']} дополнительных релевантных ответов")
    else:
        print(f"\nВЫВОД: Paraphrasing не нашел новых ответов. Возможно, нужно улучшить генерацию парафразов")
    
    print("=" * 80)
    
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / "experiment_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json_results = []
        for result_dict in all_results:
            json_results.append({
                "baseline": asdict(result_dict["baseline"]),
                "paraphrasing": asdict(result_dict["paraphrasing"])
            })
        json.dump(json_results, f, ensure_ascii=False, indent=2)
    
    print(f"\nРезультаты сохранены в: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
