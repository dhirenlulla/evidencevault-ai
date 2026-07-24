from groq import (
    APIConnectionError,
    APITimeoutError,
    APIStatusError,
    AsyncGroq
)

from app.core.config import get_settings
from app.core.exceptions import (
    LLMConnectionError,
    LLMGenerationError,
    LLMTimeoutError,
)
from app.services.llm import BaseLLMService


class  GroqLLMService(BaseLLMService):
    """ 
    Groq implementation of the BaseLLMService.
    
    This service is responsible only for sending the
    completed prompt to the configured Groq model and
    returning the generated answer.
    """
    
    def __init__(self) -> None:
        settings = get_settings()
        
        self.settings = settings
        
        self.client = AsyncGroq(
            api_key=settings.groq_api_key,
        )
        
    async def generate_answer(
        self,
        *,
        prompt: str,
    ) -> str :
        """ 
        Generate a grounded answer using the configured Groq language model.
        """
        
        try:
            response = (
                await self.client.chat.completions.create(
                    model=self.settings.groq_model_name,
                    messages=[
                        {
                            "role" : "user",
                            "content" : prompt
                        }
                    ],
                    temperature=(
                        self.settings.llm_temperature
                    ),
                    max_completion_tokens=(
                        self.settings.llm_max_tokens,
                    ),
                )
            )
            
            return (
                response.choices[0]
                .message.content
                or ""
            )
            
        except APIConnectionError as exc:
            raise LLMConnectionError(
                "Could not connect to the Groq API."
            ) from exc
            
        except APITimeoutError as exc:
            raise LLMTimeoutError(
                "The Groq request timed out."
            ) from exc
            
        except APIStatusError as exc:
            raise LLMGenerationError(
                "Groq returned an error "
                f"(status {exc.status_code})."
            ) from exc
            
        except Exception as exc:
            raise LLMGenerationError(
                "an unexpected error occurred while "
                "generating the answer."
            ) from exc