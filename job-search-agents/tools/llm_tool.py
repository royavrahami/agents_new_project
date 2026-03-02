"""
LLM Tool — thin, reusable wrapper around OpenAI's Chat Completions API.
All agents use this single interface so the underlying model can be swapped
in one place (settings.openai_model) without touching agent code.
"""

from __future__ import annotations

from typing import List, Optional

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import settings
from core.logger import logger


class LLMTool:
    """
    Wrapper around OpenAI Chat Completions.

    Args:
        model:       Model identifier (defaults to settings.openai_model)
        temperature: Sampling temperature (defaults to settings.openai_temperature)
        system_prompt: Default system prompt injected into every call
    """

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        system_prompt: str = "You are a helpful job search assistant.",
    ) -> None:
        self.model = model or settings.openai_model
        self.temperature = temperature if temperature is not None else settings.openai_temperature
        self.system_prompt = system_prompt
        self._client = OpenAI(api_key=settings.openai_api_key)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def complete(
        self,
        user_message: str,
        system_override: Optional[str] = None,
        max_tokens: int = 2000,
    ) -> str:
        """
        Send a single user message and return the assistant's text response.

        Args:
            user_message:    The user turn content
            system_override: Override the default system prompt for this call
            max_tokens:      Maximum tokens in the response

        Returns:
            Plain text response from the model

        Raises:
            openai.OpenAIError: On API failure after retries
        """
        if not settings.openai_api_key:
            logger.warning(
                "OPENAI_API_KEY not set — returning mock LLM response. "
                "Set OPENAI_API_KEY in .env to enable real LLM calls."
            )
            return f"[MOCK LLM RESPONSE for: {user_message[:80]}...]"

        messages = [
            {"role": "system", "content": system_override or self.system_prompt},
            {"role": "user", "content": user_message},
        ]

        logger.debug(f"LLM call | model={self.model} | tokens_limit={max_tokens}")

        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=max_tokens,
        )

        text = response.choices[0].message.content or ""
        logger.debug(f"LLM response length: {len(text)} chars")
        return text

    def score_relevance(self, job_description: str, candidate_profile: str) -> float:
        """
        Ask the LLM to score how relevant a job is for the candidate (0.0 – 1.0).

        Args:
            job_description:  Raw job posting text
            candidate_profile: Short text summary of candidate background

        Returns:
            Float relevance score between 0.0 and 1.0
        """
        prompt = (
            f"Rate the relevance of this job for the candidate on a scale from 0.0 to 1.0.\n\n"
            f"CANDIDATE PROFILE:\n{candidate_profile}\n\n"
            f"JOB DESCRIPTION:\n{job_description}\n\n"
            f"Respond with ONLY a decimal number between 0.0 and 1.0. Nothing else."
        )
        raw = self.complete(prompt, max_tokens=10).strip()

        try:
            score = float(raw)
            return max(0.0, min(1.0, score))
        except ValueError:
            logger.warning(f"Could not parse LLM relevance score from: {raw!r}")
            return 0.5

    def extract_company_name(self, text: str) -> Optional[str]:
        """
        Extract the primary company name mentioned in a news headline or snippet.

        Args:
            text: Headline or short article text

        Returns:
            Company name string or None
        """
        prompt = (
            f"Extract ONLY the company name from this text. "
            f"Respond with ONLY the company name, nothing else.\n\nTEXT: {text}"
        )
        result = self.complete(prompt, max_tokens=20).strip()
        return result if result and len(result) < 100 else None
