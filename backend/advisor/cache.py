import hashlib
from typing import Optional

from backend.schemas import AdvisorInput


_advisor_cache: dict[str, str] = {}  # simple in-memory cache for beta


def _cache_key(advisor_input: AdvisorInput) -> str:
    """Hash the input so identical academic states produce identical cache hits,
    regardless of when they're requested."""
    # Exclude tone/language from cache key so same academic data reuses response
    # across different tone requests (include them if you want tone-specific caching)
    payload = advisor_input.model_dump_json(exclude={"tone", "language"})
    return hashlib.sha256(payload.encode()).hexdigest()


def get_cached_response(advisor_input: AdvisorInput) -> Optional[str]:
    return _advisor_cache.get(_cache_key(advisor_input))


def set_cached_response(advisor_input: AdvisorInput, response: str) -> None:
    _advisor_cache[_cache_key(advisor_input)] = response


def clear_cache() -> None:
    """Utility for tests."""
    _advisor_cache.clear()