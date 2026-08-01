from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

class BaseLLMService(ABC):
    """ 
    Abstract interface for all language model providers.
    """
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """
        Return the configured model name.
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
        
    @abstractmethod
    def generate_answer_stream(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> AsyncIterator:
        """ 
        Stream an answer from prompt, yielding text
        fragments as the model produces them instead
        of waiting for the complete answer.
        """