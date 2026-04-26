"""
gemini_client.py - Reusable wrapper around the Google Generative AI (Gemini) API.
"""

import logging
import time
import google.generativeai as genai
from google.api_core.exceptions import GoogleAPIError
from app.config import GEMINI_API_KEY

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-2.5-flash"
MAX_RETRIES = 2
RETRY_DELAY = 2  # seconds between retries


class GeminiClient:
    """Thin wrapper around the Gemini generative AI API for clean, reusable usage."""

    def __init__(self, model: str = DEFAULT_MODEL):
        """
        Initialize the Gemini client.

        Args:
            model: Gemini model name to use. Defaults to DEFAULT_MODEL.
        """
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel(model)

    def generate(self, prompt: str) -> str:
        """
        Send a prompt to Gemini and return the text response.

        Retries up to MAX_RETRIES times on transient API errors.

        Args:
            prompt: The input prompt string.

        Returns:
            The generated text response, or an empty string on failure.
        """
        for attempt in range(1, MAX_RETRIES + 2):  # 1 initial + MAX_RETRIES retries
            try:
                response = self.model.generate_content(prompt)
                return response.text.strip()
            except GoogleAPIError as e:
                logger.warning("Gemini API error (attempt %d): %s", attempt, e)
                if attempt <= MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
            except Exception as e:
                logger.error("Unexpected error during generation: %s", e)
                break

        return ""
