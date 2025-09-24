# Minimal Bokeh stub for MFG_PDE strategic typing
# Focus on methods actually used in our visualization modules

from typing import Any

# Direct model classes (for import compatibility)
class ColorBar:
    def __init__(self, **kwargs: Any) -> None: ...

class ColumnDataSource:
    def __init__(self, **kwargs: Any) -> None: ...

class HoverTool:
    def __init__(self, **kwargs: Any) -> None: ...

class LinearColorMapper:
    def __init__(self, **kwargs: Any) -> None: ...

# Bokeh models module namespace
class Models:
    ColorBar = ColorBar
    ColumnDataSource = ColumnDataSource
    HoverTool = HoverTool
    LinearColorMapper = LinearColorMapper

    def __getattr__(self, name: str) -> Any: ...

# Lowercase alias for compatibility
models = Models()

# Bokeh plotting functions
def figure(**kwargs: Any) -> Any: ...
def show(obj: Any) -> None: ...
def output_file(filename: str, **kwargs: Any) -> None: ...

# Bokeh plotting module namespace
class Plotting:
    figure = figure
    show = show
    output_file = output_file

    def __getattr__(self, name: str) -> Any: ...

# Lowercase alias for compatibility
plotting = Plotting()

# Catch-all for any missing bokeh attributes
def __getattr__(name: str) -> Any: ...
