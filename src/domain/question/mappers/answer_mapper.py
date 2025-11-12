"""
Mappers for converting between domain entities and DTOs
"""
from typing import List
from ..entities.answer import Answer
from ..delivery.dto.dto import Answer as AnswerDTO


class AnswerMapper:
    
    @staticmethod
    def entity_to_dto(entity: Answer) -> AnswerDTO:
        """
        Convert Answer entity to AnswerDTO
        
        Args:
            entity: Answer domain entity
            
        Returns:
            AnswerDTO for data transfer
        """
        return AnswerDTO(
            id=entity.id,
            text=entity.text,
            answer_id=entity.answer_id,
            metadata=entity.metadata
        )
    
    @staticmethod
    def dto_to_entity(dto: AnswerDTO) -> Answer:
        """
        Convert AnswerDTO to Answer entity
        
        Args:
            dto: AnswerDTO from data transfer
            
        Returns:
            Answer domain entity
        """
        return Answer(
            id=dto.id,
            text=dto.text,
            answer_id=dto.answer_id,
            metadata=dto.metadata
        )
    
