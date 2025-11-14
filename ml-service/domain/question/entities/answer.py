"""
Domain entities for question domain
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class Answer:
    id: str
    text: str
    answer_id: str
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate entity after initialization"""
        if not self.id or not self.id.strip():
            raise ValueError("Answer ID cannot be empty")
        if not self.text or not self.text.strip():
            raise ValueError("Answer text cannot be empty")
        if not self.answer_id or not self.answer_id.strip():
            raise ValueError("Answer ID cannot be empty")
        
        self.id = self.id.strip()
        self.text = self.text.strip()
        self.answer_id = self.answer_id.strip()
    
    def get_text_preview(self, max_length: int = 100) -> str:
        """Get a preview of the answer text"""
        if len(self.text) <= max_length:
            return self.text
        return self.text[:max_length] + "..."
    
    def has_metadata(self) -> bool:
        """Check if answer has metadata"""
        return self.metadata is not None and len(self.metadata) > 0
