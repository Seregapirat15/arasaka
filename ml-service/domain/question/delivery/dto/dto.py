"""
Data Transfer Objects for question domain
"""
from typing import Optional, Dict, Any
from pydantic import BaseModel, field_validator


class Answer(BaseModel):
    """Answer domain entity"""
    id: str
    text: str
    answer_id: str
    metadata: Optional[Dict[str, Any]] = None


class SearchResult(BaseModel):
    """Search result domain entity"""
    answer: Answer
    score: float = 0.0
    query: str = ""
