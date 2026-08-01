import time
from uuid import UUID
from collections.abc import AsyncIterator

from app.schemas.answer import (
    AnswerResponse,
    AnswerComplete,
    AnswerToken,
)
from app.services.llm import BaseLLMService
from app.services.prompt_builder import PromptBuilder
from app.services.retrieval import RetrievalService

class GenerationService:
    """ 
    Coordinates the complete RAG workflow.
    """
    
    def __init__(
        self,
        *,
        retrieval_service: RetrievalService,
        prompt_builder: PromptBuilder,
        llm_service: BaseLLMService,
    ) -> None:
        self.retrieval_service = retrieval_service
        self.prompt_builder = prompt_builder
        self.llm_service = llm_service
        
        
    async def generate_answer(
        self,
        *,
        document_id: UUID,
        query: str,
    ) -> AnswerResponse:
        """ 
        Retrieve relevant document chunks and generate a 
        grounded answer using the configured LLM.
        """
        
        start_time = time.perf_counter()
        
        retrieval_result = (
            await self.retrieval_service.retrieve(
                document_id=document_id,
                query=query,
            )
        )
        
        system_prompt = (
            self.prompt_builder.system_prompt()
        )
        
        user_prompt = (
            self.prompt_builder.user_prompt(
                retrieval_result
            )
        )
        
        
        answer = await self.llm_service.generate_answer(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        
        generation_time = round(
            (time.perf_counter() - start_time) * 1000,
            2,
        )
        
        return AnswerResponse(
            document_id=document_id,
            query=query,
            answer=answer,
            model=self.llm_service.model_name,
            grounded=(
                    retrieval_result.total_results > 0
            ),
            total_sources=retrieval_result.total_results,
            retrieved_chunks=retrieval_result.chunks,
            generation_time_ms=generation_time,
        )
        
        
    async def generate_answer_stream(
        self,
        *,
        document_id: UUID,
        query: str,
    ) -> AsyncIterator[AnswerToken | AnswerComplete]:
        """ 
        Retrieve relevant document chunks and stream a
        grounded answer.

        Retrieval and prompt construction happen exactly as
        in generate_answer, since neither can meaningfully
        stream: the LLM cannot be prompted before the
        relevant chunks are known. Only LLM generation
        streams. Yields one AnswerToken per text fragment as
        the LLM produces it, followed by exactly one
        AnswerComplete carrying the metadata that is only
        known once generation has finished.
        """
        
        start_time = time.perf_counter()
        
        retrieval_result = (
            await self.retrieval_service.retrieve(
                document_id=document_id,
                query=query,
            )
        )
        
        system_prompt = (
            self.prompt_builder.system_prompt()
        )
        
        user_prompt = (
            self.prompt_builder.user_prompt(
                retrieval_result
            )
        )
        
        async for fragment in (
            self.llm_service.generate_answer_stream(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
        ):
            yield AnswerToken(text=fragment)

        generation_time = round(
            (time.perf_counter() - start_time) * 1000,
            2,
        )

        yield AnswerComplete(
            document_id=document_id,
            query=query,
            model=self.llm_service.model_name,
            grounded=(
                retrieval_result.total_results > 0
            ),
            total_sources=retrieval_result.total_results,
            retrieved_chunks=retrieval_result.chunks,
            generation_time_ms=generation_time,
        )