"""Groq API integration for LLM-based content formatting."""

import os
import logging
from typing import Dict, Optional

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    logging.warning("Groq package not installed. Install with: pip install groq")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GroqFormatter:
    """Wrapper for Groq API to provide LLM content generation."""

    def __init__(self, config: Dict):
        """Initialize Groq client with configuration.

        Args:
            config: Configuration dictionary containing groq_api_key and groq_model
        """
        if not GROQ_AVAILABLE:
            raise ImportError("Groq package not installed. Install with: pip install groq")

        self.config = config
        self.client = None

        # Load API key from multiple sources
        api_key = (
            config.get('groq_api_key') or
            config.get('groq', {}).get('api_key') or
            config.get('llm', {}).get('groq', {}).get('api_key') or
            os.getenv('GROQ_API_KEY')
        )

        if not api_key:
            logger.warning("No Groq API key found. Set GROQ_API_KEY environment variable or configure in config.local.yaml")
            return

        # Initialize Groq client
        self.client = Groq(api_key=api_key)

        # Get model name
        self.model = (
            config.get('groq_model') or
            config.get('groq', {}).get('model') or
            config.get('llm', {}).get('groq', {}).get('model') or
            os.getenv('GROQ_MODEL') or
            'llama-3.1-70b-versatile'  # Default model
        )

        logger.info(f"Groq API client initialized with model: {self.model}")

    def generate_content(self, prompt: str, max_tokens: int = 8000, temperature: float = 0.2) -> 'GroqResponse':
        """Generate content using Groq API.

        Args:
            prompt: The prompt to send to Groq
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0.0-2.0)

        Returns:
            GroqResponse object with text and metadata

        Raises:
            ValueError: If API key is not configured or API call fails
        """
        if not self.client:
            raise ValueError(
                "Groq API key not configured. "
                "Get a free API key at https://console.groq.com and set GROQ_API_KEY environment variable "
                "or add it to config.local.yaml"
            )

        try:
            # Call Groq API (OpenAI-compatible interface)
            completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            # Extract response text
            response_text = completion.choices[0].message.content

            # Get finish reason
            finish_reason = completion.choices[0].finish_reason

            # Create response object
            return GroqResponse(
                text=response_text,
                finish_reason=finish_reason,
                model=self.model,
                completion=completion
            )

        except Exception as e:
            error_str = str(e).lower()

            # Handle specific error types
            if "api key" in error_str or "unauthorized" in error_str or "401" in error_str:
                logger.error(f"🔑 GROQ API KEY ERROR: {e}")
                logger.error("❌ Your Groq API key is invalid or unauthorized")
                logger.error("💡 To fix this:")
                logger.error("   1. Go to https://console.groq.com")
                logger.error("   2. Sign up for free or sign in")
                logger.error("   3. Generate a new API key")
                logger.error("   4. Set GROQ_API_KEY environment variable or add to config.local.yaml")
                raise ValueError(f"Invalid Groq API key. Get a free key at https://console.groq.com")

            elif "rate limit" in error_str or "429" in error_str:
                logger.error(f"📊 GROQ RATE LIMIT ERROR: {e}")
                logger.error("❌ Groq API rate limit exceeded")
                logger.error("💡 Free tier limits:")
                logger.error("   - 30 requests per minute")
                logger.error("   - 14,400 requests per day")
                logger.error("   Wait a moment and try again")
                raise ValueError(f"Groq rate limit exceeded. Please wait and try again.")

            elif "quota" in error_str:
                logger.error(f"📊 GROQ QUOTA ERROR: {e}")
                logger.error("❌ Groq API quota has been exceeded")
                raise ValueError(f"Groq quota exceeded: {e}")

            else:
                # Unknown error
                logger.error(f"❌ Groq API error: {e}")
                raise ValueError(f"Groq API call failed: {e}")


class GroqResponse:
    """Response object mimicking Gemini's response structure for compatibility."""

    def __init__(self, text: str, finish_reason: str, model: str, completion: Optional[object] = None):
        """Initialize response object.

        Args:
            text: Generated text content
            finish_reason: Reason for completion (stop, length, etc.)
            model: Model name used
            completion: Original Groq completion object
        """
        self._text = text
        self._finish_reason = finish_reason
        self._model = model
        self._completion = completion

        # Create a mock candidates structure for compatibility with Gemini code
        self.candidates = [GroqCandidate(text, finish_reason)]

    @property
    def text(self) -> str:
        """Get the response text."""
        return self._text.strip()

    @property
    def finish_reason(self) -> str:
        """Get the finish reason."""
        return self._finish_reason

    @property
    def model(self) -> str:
        """Get the model name."""
        return self._model


class GroqCandidate:
    """Mock candidate object for Gemini compatibility."""

    def __init__(self, text: str, finish_reason: str):
        self.content = GroqContent(text)
        self.finish_reason = self._map_finish_reason(finish_reason)

    def _map_finish_reason(self, groq_finish_reason: str) -> int:
        """Map Groq finish reasons to Gemini-style codes.

        Args:
            groq_finish_reason: Groq's finish_reason string

        Returns:
            Integer code compatible with Gemini's finish_reason system
        """
        # Groq finish reasons: "stop", "length", "content_filter", etc.
        # Gemini finish reasons: 0=UNKNOWN, 1=STOP, 2=SAFETY, 3=RECITATION, 4=OTHER

        if groq_finish_reason == "stop":
            return 1  # STOP - normal completion
        elif groq_finish_reason == "length":
            return 4  # OTHER - hit token limit
        elif groq_finish_reason == "content_filter":
            return 2  # SAFETY - content filtered (rare with Groq)
        else:
            return 0  # UNKNOWN


class GroqContent:
    """Mock content object for Gemini compatibility."""

    def __init__(self, text: str):
        self.parts = [GroqPart(text)]


class GroqPart:
    """Mock part object for Gemini compatibility."""

    def __init__(self, text: str):
        self.text = text
