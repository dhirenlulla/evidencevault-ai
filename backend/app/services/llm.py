from abc import ABC, abstractmethod

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