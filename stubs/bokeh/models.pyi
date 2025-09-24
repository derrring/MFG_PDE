# Bokeh models stub for MFG_PDE strategic typing
# Focused on visualization modules used in the codebase

from typing import Any

# Model classes used in MFG_PDE visualization
class ColorBar:
    def __init__(self, **kwargs: Any) -> None: ...

class ColumnDataSource:
    def __init__(self, **kwargs: Any) -> None: ...
    def data(self) -> Any: ...

class HoverTool:
    def __init__(self, **kwargs: Any) -> None: ...

class LinearColorMapper:
    def __init__(self, **kwargs: Any) -> None: ...

# Tool classes
class BoxZoomTool:
    def __init__(self, **kwargs: Any) -> None: ...

class PanTool:
    def __init__(self, **kwargs: Any) -> None: ...

class ResetTool:
    def __init__(self, **kwargs: Any) -> None: ...

class SaveTool:
    def __init__(self, **kwargs: Any) -> None: ...

class WheelZoomTool:
    def __init__(self, **kwargs: Any) -> None: ...

# Tools submodule
class tools:
    BoxZoomTool = BoxZoomTool
    PanTool = PanTool
    ResetTool = ResetTool
    SaveTool = SaveTool
    WheelZoomTool = WheelZoomTool

# Catch-all for any missing model types
def __getattr__(name: str) -> Any: ...
