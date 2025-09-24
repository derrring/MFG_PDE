# Minimal Bokeh stub for MFG_PDE strategic typing
# Focus on methods actually used in our visualization modules

from typing import Any

# Bokeh models module
class models:
    class ColorBar:
        def __init__(self, **kwargs: Any) -> None: ...

    class ColumnDataSource:
        def __init__(self, **kwargs: Any) -> None: ...

    class HoverTool:
        def __init__(self, **kwargs: Any) -> None: ...

    class LinearColorMapper:
        def __init__(self, **kwargs: Any) -> None: ...

    # Catch-all for missing model types
    @staticmethod
    def __getattr__(name: str) -> Any: ...

# Bokeh plotting module
class plotting:
    @staticmethod
    def figure(**kwargs: Any) -> Any: ...

    @staticmethod
    def show(obj: Any) -> None: ...

    @staticmethod
    def __getattr__(name: str) -> Any: ...

# Legacy direct ColorBar for backward compatibility
class ColorBar:
    def __init__(self, **kwargs: Any) -> None: ...

# Catch-all for any missing bokeh attributes
def __getattr__(name: str) -> Any: ...
