# Minimal JAX stub for MFG_PDE strategic typing
# Focus on core JAX functions used in accelerated solvers

from collections.abc import Callable
from typing import Any

import numpy as np

# JAX Array type - simple compatibility
Array = Any

# Core JAX transformation functions
def jit(fun: Callable[..., Any], **kwargs: Any) -> Callable[..., Any]: ...
def grad(fun: Callable[..., Any], **kwargs: Any) -> Callable[..., Any]: ...
def jacfwd(fun: Callable[..., Any], **kwargs: Any) -> Callable[..., Any]: ...
def vmap(fun: Callable[..., Any], **kwargs: Any) -> Callable[..., Any]: ...

# JAX control flow
def while_loop(cond: Callable, body: Callable, init: Any) -> Any: ...

# JAX numpy compatibility
numpy = Any  # jax.numpy module

# Catch-all for any missing jax attributes
def __getattr__(name: str) -> Any: ...
