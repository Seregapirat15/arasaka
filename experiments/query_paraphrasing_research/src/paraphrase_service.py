from typing import List, Optional
from logging import getLogger
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sentence_transformers import SentenceTransformer
import numpy as np

logger = getLogger(__name__)


class ParaphraseService:
    
    def __init__(
        self,
        model_name: str = "cointegrated/rut5-base-paraphraser",
        device: str = "cpu",
        similarity_threshold: float = 0.7
    ):
        self.model_name = model_name
        self.device = device
        self.similarity_threshold = similarity_threshold
        
        try:
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)
            self.model.to(device)
            self.model.eval()
        except Exception as e:
            logger.error(f"Failed to load paraphrase model: {e}")
            raise
        
        try:
            self.similarity_model = SentenceTransformer("ai-forever/FRIDA", device=device)
        except Exception as e:
            self.similarity_model = SentenceTransformer(
                "intfloat/multilingual-e5-base",
                device=device
            )
    
    def generate_paraphrases(
        self,
        query: str,
        num_paraphrases: int = 5,
        temperature: float = 0.9,
        top_p: float = 0.95,
        max_length: int = 128
    ) -> List[str]:
        if not query or not query.strip():
            return []
        
        prefix = "paraphrase: "
        text = prefix + query
        
        try:
            inputs = self.tokenizer(
                text,
                max_length=max_length,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    num_return_sequences=num_paraphrases * 2,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    repetition_penalty=1.2
                )
            
            paraphrases = []
            for output in outputs:
                paraphrase = self.tokenizer.decode(output, skip_special_tokens=True)
                paraphrase = paraphrase.strip()
                if paraphrase and paraphrase.lower() != query.lower():
                    paraphrases.append(paraphrase)
            
            paraphrases = list(dict.fromkeys(paraphrases))
            
            filtered_paraphrases = self._filter_by_similarity(query, paraphrases)
            result = filtered_paraphrases[:num_paraphrases]
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate paraphrases: {e}")
            return []
    
    def _filter_by_similarity(
        self,
        original: str,
        candidates: List[str]
    ) -> List[str]:
        if not candidates:
            return []
        
        try:
            embeddings = self.similarity_model.encode(
                [original] + candidates,
                normalize_embeddings=True
            )
            
            original_emb = embeddings[0]
            candidate_embs = embeddings[1:]
            
            similarities = np.dot(candidate_embs, original_emb)
            
            filtered = [
                (cand, sim) for cand, sim in zip(candidates, similarities)
                if sim >= self.similarity_threshold
            ]
            filtered.sort(key=lambda x: x[1], reverse=True)
            
            return [paraphrase for paraphrase, _ in filtered]
            
        except Exception as e:
            logger.warning(f"Failed to filter by similarity: {e}")
            return candidates


class SimpleParaphraseService:
    
    def __init__(self):
        pass
    
    def generate_paraphrases(
        self,
        query: str,
        num_paraphrases: int = 3
    ) -> List[str]:
        if not query or not query.strip():
            return []
        
        paraphrases = []
        query_lower = query.lower()
        
        paraphrases.append(query)
        
        if query_lower.endswith("?"):
            paraphrases.append(query[:-1].strip())
        
        if not query_lower.endswith("?"):
            paraphrases.append(query + "?")
        
        paraphrases = [p.strip() for p in paraphrases if p.strip()]
        paraphrases = list(dict.fromkeys(paraphrases))
        
        return paraphrases[:num_paraphrases]
