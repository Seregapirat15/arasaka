"""
Paraphrase service for generating query variations
"""
from typing import List, Optional
from logging import getLogger
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sentence_transformers import SentenceTransformer
import numpy as np

logger = getLogger(__name__)


class ParaphraseService:
    """
    Service for generating paraphrases of queries using T5 model
    """
    
    def __init__(
        self,
        model_name: str = "cointegrated/rut5-base-paraphraser",
        device: str = "cpu",
        similarity_threshold: float = 0.7
    ):
        """
        Initialize paraphrase service
        
        Args:
            model_name: Name of the T5 model for paraphrasing
            device: Device to run the model on (cpu/cuda)
            similarity_threshold: Minimum similarity score for filtering paraphrases
        """
        self.model_name = model_name
        self.device = device
        self.similarity_threshold = similarity_threshold
        
        try:
            logger.info(f"Loading paraphrase model: {model_name}")
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            
            # Загружаем модель с явным указанием устройства, чтобы избежать проблем с meta-тензорами
            # Отключаем все автоматические оптимизации, которые могут использовать meta-устройство
            torch_device = torch.device(device)
            
            # Загружаем модель с параметрами, которые гарантируют загрузку на CPU без meta-тензоров
            # Используем device_map=None и low_cpu_mem_usage=False
            try:
                self.model = T5ForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=False,
                    device_map=None  # Явно отключаем device_map
                )
            except TypeError:
                # Если device_map не поддерживается в старой версии transformers
                self.model = T5ForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=False
                )
            
            # Проверяем, что модель не на meta-устройстве
            try:
                first_param = next(self.model.parameters())
                if hasattr(first_param, 'device') and first_param.device.type == 'meta':
                    raise RuntimeError("Model loaded on meta device")
            except (StopIteration, AttributeError):
                pass
            
            # Перемещаем модель на нужное устройство
            self.model = self.model.to(torch_device)
            self.model.eval()
            logger.info(f"Paraphrase model loaded successfully on {device}")
        except Exception as e:
            error_msg = str(e).lower()
            logger.error(f"Failed to load paraphrase model: {e}")
            
            # Если ошибка связана с meta-тензорами, пробуем альтернативный способ
            if "meta tensor" in error_msg or "to_empty" in error_msg or "meta device" in error_msg:
                logger.warning("Meta tensor error detected, trying alternative load method")
                try:
                    # Пробуем загрузить модель без всех оптимизаций
                    import os
                    # Временно отключаем accelerate, если он используется
                    os.environ['ACCELERATE_DISABLE_RICH'] = '1'
                    
                    self.model = T5ForConditionalGeneration.from_pretrained(
                        model_name,
                        torch_dtype=torch.float32
                    )
                    # Явно перемещаем на CPU сначала, затем на нужное устройство
                    self.model = self.model.cpu()
                    if device != 'cpu':
                        self.model = self.model.to(torch.device(device))
                    self.model.eval()
                    logger.info(f"Paraphrase model loaded successfully on {device} (alternative method)")
                except Exception as e2:
                    logger.error(f"Alternative load method also failed: {e2}")
                    raise
            else:
                raise
        
        try:
            # Try to use the same embedding model as the main service
            self.similarity_model = SentenceTransformer("ai-forever/FRIDA", device=device)
            logger.info("Using FRIDA for similarity filtering")
        except Exception as e:
            logger.warning(f"Failed to load FRIDA, using fallback: {e}")
            self.similarity_model = SentenceTransformer(
                "intfloat/multilingual-e5-base",
                device=device
            )
            logger.info("Using multilingual-e5-base for similarity filtering")
    
    def generate_paraphrases(
        self,
        query: str,
        num_paraphrases: int = 5,
        temperature: float = 0.9,
        top_p: float = 0.95,
        max_length: int = 128
    ) -> List[str]:
        """
        Generate paraphrases for a given query
        
        Args:
            query: Original query text
            num_paraphrases: Number of paraphrases to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_length: Maximum length of generated text
            
        Returns:
            List of paraphrased queries
        """
        if not query or not query.strip():
            return []
        
        prefix = "paraphrase: "
        text = prefix + query
        
        try:
            # Токенизируем текст
            tokenized = self.tokenizer(
                text,
                max_length=max_length,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            
            # Перемещаем на нужное устройство, используя torch.device
            torch_device = torch.device(self.device)
            inputs = {
                "input_ids": tokenized["input_ids"].to(torch_device),
                "attention_mask": tokenized["attention_mask"].to(torch_device)
            }
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    num_return_sequences=num_paraphrases * 2,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    repetition_penalty=1.2,
                    pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
                )
            
            paraphrases = []
            for output in outputs:
                paraphrase = self.tokenizer.decode(output, skip_special_tokens=True)
                paraphrase = paraphrase.strip()
                if paraphrase and paraphrase.lower() != query.lower():
                    paraphrases.append(paraphrase)
            
            # Remove duplicates while preserving order
            paraphrases = list(dict.fromkeys(paraphrases))
            
            # Filter by similarity to original query
            filtered_paraphrases = self._filter_by_similarity(query, paraphrases)
            result = filtered_paraphrases[:num_paraphrases]
            
            logger.debug(f"Generated {len(result)} paraphrases for query: {query[:50]}...")
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate paraphrases: {e}")
            return []
    
    def _filter_by_similarity(
        self,
        original: str,
        candidates: List[str]
    ) -> List[str]:
        """
        Filter paraphrases by similarity to original query
        
        Args:
            original: Original query
            candidates: Candidate paraphrases
            
        Returns:
            Filtered list of paraphrases sorted by similarity
        """
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
    """
    Simple paraphrase service that returns the original query with minor variations
    Used as fallback when T5 model is not available
    """
    
    def __init__(self):
        pass
    
    def generate_paraphrases(
        self,
        query: str,
        num_paraphrases: int = 3
    ) -> List[str]:
        """
        Generate simple paraphrases (just adds/removes question mark)
        
        Args:
            query: Original query text
            num_paraphrases: Number of paraphrases to generate
            
        Returns:
            List of simple paraphrases
        """
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

