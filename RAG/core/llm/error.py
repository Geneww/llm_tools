import logging
from functools import wraps
from typing import Optional
import openai


class LLMError(Exception):
    """Base class for all LLM exceptions."""
    description: Optional[str] = None

    def __init__(self, description: Optional[str] = None) -> None:
        self.description = description


class LLMBadRequestError(LLMError):
    """Raised when the LLM returns bad request."""
    description = "Bad Request"


class LLMAPIConnectionError(LLMError):
    """Raised when the LLM returns API connection error."""
    description = "API Connection Error"


class LLMAPIUnavailableError(LLMError):
    """Raised when the LLM returns API unavailable error."""
    description = "API Unavailable Error"


class LLMRateLimitError(LLMError):
    """Raised when the LLM returns rate limit error."""
    description = "Rate Limit Error"


class LLMAuthorizationError(LLMError):
    """Raised when the LLM returns authorization error."""
    description = "Authorization Error"


class ProviderTokenNotInitError(Exception):
    """
    Custom exception raised when the provider token is not initialized.
    """
    description = "Provider Token Not Init"

    def __init__(self, *args, **kwargs):
        self.description = args[0] if args else self.description


class QuotaExceededError(Exception):
    """
    Custom exception raised when the quota for a provider has been exceeded.
    """
    description = "Quota Exceeded"


class ModelCurrentlyNotSupportError(Exception):
    """
    Custom exception raised when the model not support
    """
    description = "Model Currently Not Support"


def handle_llm_exceptions(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except openai.BadRequestError as e:
            logging.exception("Invalid request to OpenAI API.")
            raise LLMBadRequestError(str(e))
        except openai.APIConnectionError as e:
            logging.exception("Failed to connect to OpenAI API.")
            raise LLMAPIConnectionError(str(e))
        except (openai.APIError, openai.Timeout) as e:
            logging.exception("OpenAI service unavailable.")
            raise LLMAPIUnavailableError(str(e))
        except openai.RateLimitError as e:
            raise LLMRateLimitError(str(e))
        except openai.AuthenticationError as e:
            raise LLMAuthorizationError(str(e))
        except Exception as e:
            raise Exception(e)

    return wrapper
