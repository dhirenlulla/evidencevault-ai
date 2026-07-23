from app.schemas.retrieval import RetrievalResult


class PromptBuilder:
    """
    Builds grounded prompts for Retrieval-Augmented Generation.
    """

    SYSTEM_PROMPT = """
You are EvidenceVault AI, an enterprise document intelligence assistant.

Answer ONLY using the supplied document context.

Rules:

1. Do not use outside knowledge.

2. If the answer is not contained in the context,
   say that the document does not contain enough
   information.

3. Do not fabricate facts.

4. Be concise and accurate.

5. When possible, mention the page number naturally
   in your answer.
""".strip()

    def build_prompt(
        self,
        retrieval_result: RetrievalResult,
    ) -> str:
        """
        Construct the complete prompt for the LLM.
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
{self.SYSTEM_PROMPT}

------------------------
Document Context
------------------------

{context}

------------------------
User Question
------------------------

{retrieval_result.query}

------------------------
Answer
------------------------
""".strip()