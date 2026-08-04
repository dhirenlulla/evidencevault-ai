import time
from collections.abc import AsyncIterator
from uuid import UUID

from app.schemas.answer import (
    AnswerComplete,
    AnswerResponse,
    AnswerToken,
)
from app.schemas.retrieval import RetrievalResult
from app.services.hybrid_retrieval import (
    HybridRetrievalService,
)
from app.services.llm import BaseLLMService
from app.services.prompt_builder import PromptBuilder
from app.services.reranker import CrossEncoderReranker

class RAGPipelineService:
    """
    Orchestrates the complete retrieve -> fuse -> rerank -> 
    generate pipeline.
    
    This service intentionally contains almost no logic of its own.
    It composes four services that were each built and tested in isolation.
    
    HybridRetrievalService -- dense + BM25 retrieval, fused with
                              RRF
    CrossEncoderReranker   -- reorders the fused shortlist for higher-
                              precision context.
    PromptBuilder          -- constructs the grounded prompt
    BaseLLMService         -- generates the final answer
    
    Its job is only to pass the right shape of data between them 
    (for example, wrapping reranked chunks back into a RetrievalResult
    so PromptBuilder can be reused unchanged) in the correct order. This
    mirrors GenerationService's role but wires the richer, slower,
    higher-quality retrieval pipeline instead of dense-only retrieval.
    """
    
    def __init__(
        self,
        *,
        hybrid_retrieval_service: HybridRetrievalService,
        reranker: CrossEncoderReranker,
        prompt_builder: PromptBuilder,
        llm_service: BaseLLMService,
    )-> None:
        self.hybrid_retrieval_service = (
            hybrid_retrieval_service
        ) 
        self.reranker = reranker
        self.prompt_builder = prompt_builder
        self.llm_service = llm_service
        
    async def _retrieve_and_rerank(
        self,
        *,
        document_id: UUID,
        query: str,
    ) -> RetrievalResult:
        """ 
        Run hybrid retrieval, rerank its fused results, and return
        them as a RetrievalResult.
        
        Returning a RetrievalResult (rather than a bare tuple of 
        reranked chunks) lets PromptBuilder be reused completely 
        unchanged, since it only ever reads .chunks and .query from 
        whatever it's given.
        """
        
        hybrid_result = (
            await self.hybrid_retrieval_service.retrieve(
                document_id=document_id,
                query=query,
            )
        )
        
        reranked = await self.reranker.rerank_async(
            query=query,
            chunks=hybrid_result.fused_results,
        )
        
        reranked_chunks = tuple(
            item.chunk for item in reranked
        )
        
        return RetrievalResult(
            query=query,
            total_results=len(reranked_chunks),
            chunks=reranked_chunks,
        )
        
    async def generate_answer(
        self,
        *,
        document_id: UUID,
        query: str,
    ) -> AnswerResponse:
        """ 
        Run the complete pipeline and return one complete answer,
        once generation has finished.
        """
        
        start_time = time.perf_counter()
        
        retrieval_result = (
            await self._retrieve_and_rerank(
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
            total_sources=(
                retrieval_result.total_results
            ),
            retrieved_chunks=retrieval_result.chunks,
            generation_time_ms=generation_time,
        )
        
    async def generate_answer_stream(
        self,
        *,
        document_id: UUID,
        query:str,
    ) -> AsyncIterator[AnswerToken | AnswerComplete]:
        """ 
        Run the complete pipeline and stream the answer, yielding
        AnswerToken fragments followed by exactly one final 
        AnswerComplete.
        """
        
        start_time = time.perf_counter()
        
        retrieval_result = (
            await self._retrieve_and_rerank(
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
            total_sources=(
                retrieval_result.total_results
            ),
            retrieved_chunks=retrieval_result.chunks,
            generation_time_ms=generation_time,
        )