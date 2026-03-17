"""Generation pipeline: prompt build → LLM call → format answer."""

from typing import Literal

from app.core.config import Settings
from app.core.logging import get_logger
from app.core.schemas import Citation, RetrievalResult
from app.generation.citation_formatter import CitationFormatter
from app.generation.generator import BaseLLMProvider, create_llm_provider
from app.generation.prompt_builder import PromptBuilder

logger = get_logger(__name__)


class GenerationPipeline:
    """Orchestrates prompt construction, LLM generation, and citation formatting."""

    def __init__(self, settings: Settings, llm_provider: BaseLLMProvider | None = None):
        self.settings = settings
        self.prompt_builder = PromptBuilder()
        self.citation_formatter = CitationFormatter()
        self.llm = llm_provider or create_llm_provider(settings)

    def run(
        self,
        retrieval_result: RetrievalResult,
        mode: Literal["concise", "analysis"] = "concise",
    ) -> tuple[str, list[Citation]]:
        """Generate an answer from retrieval results.

        Returns:
            (answer_text, citations)
        """
        system_prompt, user_prompt = self.prompt_builder.build(
            query=retrieval_result.query,
            context=retrieval_result.context_text,
            mode=mode,
        )

        raw_answer = self.llm.generate(system_prompt, user_prompt)

        # Append reference list
        ref_block = self.citation_formatter.format_references(retrieval_result.citations)
        answer = f"{raw_answer}\n\n### References\n{ref_block}"

        logger.info("Generation complete: %d chars answer, %d citations", len(answer), len(retrieval_result.citations))
        return answer, retrieval_result.citations
