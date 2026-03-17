"""Prompt templates for RAG generation."""

from typing import Literal

SYSTEM_PROMPT = """You are a helpful AI research assistant specializing in academic papers.
You answer questions based ONLY on the provided retrieved paper excerpts.

Rules:
1. Only use information from the provided context to answer the question.
2. If the context does not contain sufficient information, say "I don't have enough information in the retrieved papers to answer this question."
3. Never fabricate paper conclusions, results, or claims.
4. Cite sources by referencing the paper number [1], [2], etc.
5. When comparing topics, synthesize information across multiple papers.
6. Be precise about what each paper actually says vs. your interpretation."""

CONCISE_TEMPLATE = """Based on the following retrieved paper excerpts, answer the question concisely.

## Retrieved Papers
{context}

## Question
{query}

## Instructions
- Answer in 2-4 sentences
- Cite relevant papers using [1], [2], etc.
- If the papers don't cover the question, say so clearly"""

ANALYSIS_TEMPLATE = """Based on the following retrieved paper excerpts, provide a detailed analysis.

## Retrieved Papers
{context}

## Question
{query}

## Instructions
- Provide a thorough analysis drawing from multiple papers
- Compare and contrast different approaches or findings when applicable
- Cite specific papers using [1], [2], etc.
- Structure your answer with clear points
- If the papers don't fully cover the question, note the gaps"""


class PromptBuilder:
    """Builds prompts for the LLM."""

    def build(
        self,
        query: str,
        context: str,
        mode: Literal["concise", "analysis"] = "concise",
    ) -> tuple[str, str]:
        """Return (system_prompt, user_prompt)."""
        template = CONCISE_TEMPLATE if mode == "concise" else ANALYSIS_TEMPLATE
        user_prompt = template.format(context=context, query=query)
        return SYSTEM_PROMPT, user_prompt
