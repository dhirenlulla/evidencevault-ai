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
    LLMAuthenticationError,
    LLMRateLimitError,
)
from app.services.llm import BaseLLMService


class GroqLLMService(BaseLLMService):
    """ 
    Groq implementation of the BaseLLMService.
    
    This service is responsible only for sending the
    completed prompt to the configured Groq model and
    returning the generated answer.
    """
    
    def __init__(self) -> None:
        settings = get_settings()
        
        self._settings = settings
        
        self.client = AsyncGroq(
            api_key=settings.groq_api_key,
            timeout=settings.llm_timeout_seconds,
        )
        
    
    @property
    def model_name(self) -> str:
        """ 
        Return the configured model name.
        """
        
        return self._settings.groq_model_name
    
    
    async def generate_answer(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> str :
        """ 
        Generate a grounded answer using the configured Groq language model.
        """
        
        try:
            response = (
                await self.client.chat.completions.create(
                    model=self._settings.groq_model_name,
                    messages=[
                        {
                            "role" : "system",
                            "content" : system_prompt,
                        },
                        {
                            "role" : "user",
                            "content" : user_prompt,
                        }
                    ],
                    temperature=(
                        self._settings.llm_temperature
                    ),
                    max_completion_tokens=(
                        self._settings.llm_max_tokens
                    ),
                )
            )
            
            if not response.choices:
                raise LLMGenerationError(
                    "Groq returned no completion choices."
                )
            
            answer = (
                response.choices[0]
                .message.content
            )
            
            if not answer:
                raise LLMGenerationError(
                    "Groq returned an empty response."
                )
                
            return answer.strip()
            
        except APIConnectionError as exc:
            raise LLMConnectionError(
                "Could not connect to the Groq API."
            ) from exc
            
        except APITimeoutError as exc:
            raise LLMTimeoutError(
                "The Groq request timed out."
            ) from exc
            
        except APIStatusError as exc:
            
            if exc.status_code == 401:
                raise LLMAuthenticationError(
                    "Invalid Groq API key."
                ) from exc
                
            if exc.status_code == 429:
                raise LLMRateLimitError(
                    "Groq rate limit exceeded."
                ) from exc
                
            raise LLMGenerationError(
                "Groq returned an error "
                f"(status {exc.status_code})."
            ) from exc
            
        except (
            LLMConnectionError,
            LLMTimeoutError,
            LLMGenerationError,
        ):
            raise
        
        except Exception as exc:
            raise LLMGenerationError(
                "An unexpected error occurred while "
                "generating the answer."
            ) from exc
            