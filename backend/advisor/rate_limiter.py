import time
from collections import deque
from threading import Lock


class RateLimiter:
    """Simple sliding-window rate limiter. Thread-safe for a single-process app."""
    
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._timestamps: deque[float] = deque()
        self._lock = Lock()

    def allow_request(self) -> bool:
        now = time.time()
        with self._lock:
            # Drop timestamps outside the window
            while self._timestamps and self._timestamps[0] < now - self.window_seconds:
                self._timestamps.popleft()
            if len(self._timestamps) < self.max_requests:
                self._timestamps.append(now)
                return True
            return False


# Example: Gemini free tier often allows ~15 requests/minute
# (check current limits before hardcoding)
advisor_rate_limiter = RateLimiter(max_requests=15, window_seconds=60)