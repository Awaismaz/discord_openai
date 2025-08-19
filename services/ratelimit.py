# services/ratelimit.py
import os, time
from collections import deque
from typing import Deque, Dict, Tuple

_LIMIT = int(os.getenv("RATE_LIMIT_PER_MINUTE", "20"))
_WINDOW = 60.0  # seconds
# key: (user_id, mode) -> deque[timestamps]
_BUCKETS: Dict[Tuple[str, str], Deque[float]] = {}

def allow(user_id: str, mode: str) -> Tuple[bool, int]:
    """
    Returns (allowed, remaining_in_window)
    """
    now = time.time()
    key = (user_id, mode)
    dq = _BUCKETS.setdefault(key, deque())
    # drop old timestamps
    while dq and now - dq[0] > _WINDOW:
        dq.popleft()
    if len(dq) >= _LIMIT:
        # remaining = 0 within window
        return (False, 0)
    dq.append(now)
    remaining = max(0, _LIMIT - len(dq))
    return (True, remaining)

def reset_user(user_id: str, mode: str | None = None):
    if mode is None:
        # wipe all modes for user
        for k in [k for k in _BUCKETS.keys() if k[0] == user_id]:
            _BUCKETS.pop(k, None)
    else:
        _BUCKETS.pop((user_id, mode), None)
