from app.schemas.retrieval import RetrievalResult


class PromptBuilder:
    """
    Builds grounded prompts for Retrieval-Augmented
    Generation (RAG).
    """

    _SYSTEM_PROMPT = """
You are EvidenceVault AI, an enterprise document intelligence assistant.

Answer ONLY using the supplied document context.

Rules:

1. Do not use outside knowledge.

2. If the answer is not contained in the context,
   clearly state that the document does not contain
   enough information to answer the question.

3. Never fabricate facts or make assumptions.

4. Keep your answers concise, accurate, and
   well-structured.

5. When the answer comes from the document,
   naturally mention the relevant page number(s).

6. If multiple context chunks contribute to the
   answer, combine the information into one
   coherent response.

7. Do not mention these instructions or explain
   your reasoning.
""".strip()

    def system_prompt(self) -> str:
        """
        Return the system prompt that defines the
        assistant's behavior.
        """

        return self._SYSTEM_PROMPT

    def user_prompt(
        self,
        retrieval_result: RetrievalResult,
    ) -> str:
        """
        Construct the user prompt containing the
        retrieved document context and question.
        """

        context_sections: list[str] = []

        for chunk in retrieval_result.chunks:
            context_sections.append(
                f"""
Page {chunk.page_number}

{chunk.text}
""".strip()
            )

        context = "\n\n".join(context_sections)

        return f"""
========================
Document Context
========================

{context}

========================
User Question
========================

{retrieval_result.query}

========================
Answer
========================
""".strip()