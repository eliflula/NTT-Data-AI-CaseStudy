from __future__ import annotations

# import re  # CoT aktif edilince açılacak
from dataclasses import dataclass

from groq import Groq
from qdrant_client.models import ScoredPoint

from src.config import settings

# ---------------------------------------------------------------------------
# Chain-of-Thought eklentisi — şimdilik devre dışı
# Aktif etmek için:
#   1. Üstteki `import re` satırını uncomment yap
#   2. _COT_BLOCK'u _SYSTEM_PROMPT içindeki ## RULES satırından önce ekle
#   3. generate() metodundaki _THINKING_RE.sub satırını uncomment yap
#
# _COT_BLOCK = """
# ## REASONING PROCESS (mandatory)
#
# Before writing your final answer, reason step by step inside <thinking> tags.
# In your thinking block:
# - Identify which chunks are relevant to the question.
# - Note the years and sources of relevant data.
# - Check for numerical values; verify units and magnitudes.
# - For multi-year questions, compare figures explicitly (e.g. 2022: X → 2024: Y, delta: Z).
# - Decide if the context is sufficient to answer; if not, prepare the fallback phrase.
# - Plan a detailed response: what background info, what key figures, what trends to include.
#
# Only after completing your thinking block, write the final answer outside the tags.
# """
# _THINKING_RE = re.compile(r"<thinking>.*?</thinking>", re.DOTALL)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are an AI assistant designed to answer questions based strictly on \
retrieved context from NTT DATA sustainability and corporate documents or web search results.

## RULES

1. CONTEXT USAGE (CRITICAL)
   - Only use the provided context to generate answers.
   - Do NOT use prior knowledge.
   - If context comes from web search results (marked with [Web Search Results]):
     summarize what the results say and guide the user to the relevant sources.
     Never say "not available" when web results are present — always extract value from them.
   - If context comes from documents and the answer is not clearly present, say:
     "The information is not available in the provided documents."

2. MULTI-DOCUMENT REASONING
   - Documents may belong to different years; pay attention to year, page, and source metadata.
   - When multiple documents contain relevant information, compare them and highlight
     differences across years where relevant.

3. ANSWER QUALITY
   - Be clear, structured, and professional.
   - Provide a comprehensive and detailed answer — do not truncate relevant information.
   - Include all supporting figures, context, and explanations available in the context.
   - Use bullet points or numbered lists when presenting multiple data points or steps.

4. NO HALLUCINATION
   - Never generate assumptions or fill missing gaps with guesses.

5. FORMAT
   - Write your answer as flowing prose or bullets — do NOT use labels like
     "Direct Answer:", "Supporting Details:", or any similar header prefixes.
   - Lead with the core answer immediately, then add supporting details naturally.

6. CITATIONS
   - Do NOT include any source citations, file names, chunk labels, or page references in your answer.
   - Never mention "Chunk 1", "Chunk 2", source file names, or page numbers.

7. NUMERICAL DATA
   - Preserve exact numbers, percentages, and metrics; do not approximate.

8. STRUCTURED DATA
   - If the context contains tables or structured data, preserve the structure in output.

Your goal is to provide accurate, grounded, and verifiable answers."""

@dataclass
class GenerateResult:
    """Holds the generated answer and token usage from the LLM."""

    answer: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class GeneratorService:
    """Builds context from retrieved chunks and generates answers via an LLM."""

    def __init__(self) -> None:
        self._client = Groq(api_key=settings.groq_api_key)

    def build_context(self, results: list[ScoredPoint]) -> str:
        """Convert a list of scored chunks into a formatted context string."""
        parts: list[str] = []
        for i, point in enumerate(results, 1):
            p = point.payload or {}
            parts.append(
                f"[Chunk {i} | Source: {p.get('source', '')} | "
                f"Year: {p.get('year', '')} | Page: {p.get('page', '')} | "
                f"Score: {point.score:.3f}]\n{p.get('content', '')}"
            )
        return "\n\n---\n\n".join(parts)

    def generate(self, question: str, context: str) -> GenerateResult:
        """Generate an answer for the given question using the provided context."""
        response = self._client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
            temperature=0.2,
            max_tokens=2048,
        )
        usage = response.usage
        # answer = _THINKING_RE.sub("", raw).strip()  # CoT aktif edilince uncomment
        answer = response.choices[0].message.content
        return GenerateResult(
            answer=answer,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
        )
