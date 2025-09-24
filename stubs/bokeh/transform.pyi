# Bokeh transform stub for MFG_PDE strategic typing

from typing import Any

# Transform functions used in MFG_PDE
def linear_cmap(field_name: str, palette: Any, low: float, high: float, **kwargs: Any) -> Any: ...
def log_cmap(field_name: str, palette: Any, low: float, high: float, **kwargs: Any) -> Any: ...

# Catch-all for any missing transform functions
def __getattr__(name: str) -> Any: ...
