from abc import ABC, abstractmethod

from app.schemas.answer import AnswerResponse


class BaseLLMService(ABC):
    """ 
    Abstract interface for all language model providers.
    """
    
    @abstractmethod
    async def generate_answer(
        self,
        *,
        system_prompt: str,
        user_prompt:str,
    ) -> str:
        """ 
        Generate an answer from a prompt.
        """
        
        raise NotImplementedError