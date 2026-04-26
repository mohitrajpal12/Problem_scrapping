"""
problem_analyzer.py - Analyzes raw web content and extracts structured problem insights using Gemini.
"""

import json
import logging
import re

from app.ai.gemini_client import GeminiClient

logger = logging.getLogger(__name__)

MAX_INPUT_LENGTH = 3000

FALLBACK_STRUCTURE = {
    "problem_summary": "",
    "who_faces_it": "",
    "root_cause": "",
    "existing_solutions": "",
    "gaps_in_solutions": "",
    "business_opportunity": "",
    "solution_ideas": [],
}

PROMPT_TEMPLATE = """You are a sharp business analyst and startup strategist.

Analyze the following web content and extract structured problem intelligence.
Focus only on REAL, SPECIFIC problems — avoid generic or vague statements.
Think about who suffers, why existing solutions fail, and what monetizable opportunity exists.

Respond ONLY with a valid JSON object. No explanation, no markdown, no code blocks.

Required JSON format:
{{
  "problem_summary": "A concise 1-2 sentence description of the core problem",
  "who_faces_it": "Specific group of people or businesses affected",
  "root_cause": "The underlying reason this problem exists",
  "existing_solutions": "Current tools or approaches people use to solve it",
  "gaps_in_solutions": "What those solutions fail to address",
  "business_opportunity": "A practical, monetizable opportunity this problem presents",
  "solution_ideas": ["Idea 1", "Idea 2", "Idea 3"]
}}

Web Content:
{content}"""


class ProblemAnalyzer:
    """Extracts structured problem insights from raw web content using Gemini."""

    def __init__(self, client: GeminiClient):
        """
        Args:
            client: An initialized GeminiClient instance.
        """
        self.client = client

    def analyze(self, content: str) -> dict:
        """
        Analyze raw web content and return structured problem intelligence.

        Sends a business-analyst-style prompt to Gemini and parses the JSON response.
        Returns a fallback structure if the LLM response is invalid or an error occurs.

        Args:
            content: Raw scraped text from a webpage.

        Returns:
            A dict with keys: problem_summary, who_faces_it, root_cause,
            existing_solutions, gaps_in_solutions, business_opportunity, solution_ideas.
        """
        trimmed = content.strip()[:MAX_INPUT_LENGTH]
        if not trimmed:
            logger.warning("Empty content passed to analyzer.")
            return FALLBACK_STRUCTURE.copy()

        prompt = PROMPT_TEMPLATE.format(content=trimmed)

        try:
            raw = self.client.generate(prompt)
            return self._parse_response(raw)
        except Exception as e:
            logger.error("Analysis failed: %s", e)
            return FALLBACK_STRUCTURE.copy()

    def _parse_response(self, raw: str) -> dict:
        """
        Parse and validate the LLM's JSON response.

        Strips markdown code fences if present, then attempts JSON parsing.
        Falls back to the default structure for any missing or invalid fields.

        Args:
            raw: Raw string response from Gemini.

        Returns:
            A validated and sanitized dict matching the expected structure.
        """
        # Strip markdown code fences if Gemini wraps response in ```json ... ```
        cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON from LLM response. Raw: %s", raw[:300])
            return FALLBACK_STRUCTURE.copy()

        result = FALLBACK_STRUCTURE.copy()
        for key in result:
            value = data.get(key, result[key])
            # Ensure solution_ideas is always a list of strings
            if key == "solution_ideas":
                result[key] = [str(i) for i in value] if isinstance(value, list) else []
            else:
                result[key] = str(value) if value else ""

        return result
