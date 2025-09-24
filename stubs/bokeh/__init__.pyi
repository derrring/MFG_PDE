# Minimal Bokeh stub for MFG_PDE strategic typing
# Focus on methods actually used in our visualization modules

from typing import Any

class ColorBar:
    def __init__(self, **kwargs: Any) -> None: ...

# Catch-all for any missing bokeh attributes
def __getattr__(name: str) -> Any: ...
