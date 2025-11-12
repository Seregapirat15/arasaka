from typing import List, Dict, Tuple
from collections import defaultdict
from logging import getLogger
import numpy as np

logger = getLogger(__name__)


class AnswerResult:
    def __init__(self, answer_id: str, answer_text: str, score: float):
        self.answer_id = answer_id
        self.answer_text = answer_text
        self.score = score


class VotingRanker:
    
    def __init__(self, voting_method: str = "weighted"):
        self.voting_method = voting_method
    
    def rank_answers(
        self,
        search_results_by_paraphrase: Dict[str, List[AnswerResult]]
    ) -> List[Tuple[str, str, float, float, float]]:
        if not search_results_by_paraphrase:
            return []
        
        if self.voting_method == "simple":
            return self._simple_voting(search_results_by_paraphrase)
        elif self.voting_method == "weighted":
            return self._weighted_voting(search_results_by_paraphrase)
        elif self.voting_method == "ensemble":
            return self._ensemble_voting(search_results_by_paraphrase)
        else:
            raise ValueError(f"Unknown voting method: {self.voting_method}")
    
    def _simple_voting(
        self,
        search_results_by_paraphrase: Dict[str, List[AnswerResult]]
    ) -> List[Tuple[str, str, float, float, float]]:
        answer_data = defaultdict(lambda: {"votes": 0, "positions": []})
        answer_texts = {}
        
        for paraphrase, results in search_results_by_paraphrase.items():
            for position, result in enumerate(results, start=1):
                answer_id = result.answer_id
                answer_data[answer_id]["votes"] += 1
                answer_data[answer_id]["positions"].append(position)
                if answer_id not in answer_texts:
                    answer_texts[answer_id] = result.answer_text
        
        ranked = []
        for answer_id, data in answer_data.items():
            votes = data["votes"]
            positions = data["positions"]
            avg_position = np.mean(positions)
            position_bonus = 1.0 / avg_position
            
            final_score = votes + position_bonus
            
            ranked.append((
                answer_id,
                answer_texts[answer_id],
                float(final_score),
                float(final_score),
                float(final_score)
            ))
        
        ranked.sort(key=lambda x: x[4], reverse=True)
        
        return ranked
    
    def _weighted_voting(
        self,
        search_results_by_paraphrase: Dict[str, List[AnswerResult]]
    ) -> List[Tuple[str, str, float, float]]:
        answer_data = defaultdict(lambda: {"scores": [], "positions": [], "count": 0})
        answer_texts = {}
        
        for paraphrase, results in search_results_by_paraphrase.items():
            for position, result in enumerate(results, start=1):
                answer_id = result.answer_id
                answer_data[answer_id]["scores"].append(result.score)
                answer_data[answer_id]["positions"].append(position)
                answer_data[answer_id]["count"] += 1
                if answer_id not in answer_texts:
                    answer_texts[answer_id] = result.answer_text
        
        ranked = []
        for answer_id, data in answer_data.items():
            scores = data["scores"]
            positions = data["positions"]
            count = data["count"]
            
            avg_score = np.mean(scores)
            max_score = np.max(scores) if scores else 0.0
            position_weights = [1.0 / pos for pos in positions]
            avg_position_weight = np.mean(position_weights)
            count_bonus = count / len(search_results_by_paraphrase)
            
            ranking_score = (avg_score * 0.4) + (avg_position_weight * 0.4) + (count_bonus * 0.2)
            
            ranked.append((
                answer_id,
                answer_texts[answer_id],
                float(avg_score),
                float(max_score),
                float(ranking_score)
            ))
        
        ranked.sort(key=lambda x: x[4], reverse=True)
        
        return ranked
    
    def _ensemble_voting(
        self,
        search_results_by_paraphrase: Dict[str, List[AnswerResult]]
    ) -> List[Tuple[str, str, float, float, float]]:
        simple_results = self._simple_voting(search_results_by_paraphrase)
        weighted_results = self._weighted_voting(search_results_by_paraphrase)
        
        simple_dict = {aid: score for aid, _, score, _, _ in simple_results}
        weighted_dict = {aid: ranking_score for aid, _, _, _, ranking_score in weighted_results}
        max_score_dict = {aid: max_score for aid, _, _, max_score, _ in weighted_results}
        answer_texts = {aid: text for aid, text, _, _, _ in weighted_results}
        
        if simple_dict:
            max_simple = max(simple_dict.values())
            simple_dict = {k: v / max_simple if max_simple > 0 else 0 
                          for k, v in simple_dict.items()}
        
        if weighted_dict:
            max_weighted = max(weighted_dict.values())
            weighted_dict = {k: v / max_weighted if max_weighted > 0 else 0 
                           for k, v in weighted_dict.items()}
        
        ensemble_scores = {}
        all_answer_ids = set(simple_dict.keys()) | set(weighted_dict.keys())
        
        for answer_id in all_answer_ids:
            simple_score = simple_dict.get(answer_id, 0)
            weighted_score = weighted_dict.get(answer_id, 0)
            ensemble_score = 0.4 * simple_score + 0.6 * weighted_score
            ensemble_scores[answer_id] = ensemble_score
        
        ranked = []
        for answer_id, score in sorted(
            ensemble_scores.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            avg_score = simple_dict.get(answer_id, 0)
            max_score = max_score_dict.get(answer_id, avg_score)
            ranked.append((
                answer_id,
                answer_texts.get(answer_id, ""),
                float(avg_score),
                float(max_score),
                float(score)
            ))
        
        return ranked
