# Minimal memory_profiler stub for MFG_PDE strategic typing
# Focus on memory monitoring methods used in performance analysis

from collections.abc import Callable
from typing import Any

def profile(func: Callable[..., Any] | None = None, **kwargs: Any) -> Any: ...
def memory_usage(proc: Any = None, **kwargs: Any) -> list[float]: ...

# Catch-all for any missing memory_profiler attributes
def __getattr__(name: str) -> Any: ...
