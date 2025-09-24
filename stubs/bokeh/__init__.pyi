# Minimal Bokeh stub for MFG_PDE strategic typing
# Properly structured module hierarchy for import compatibility

from typing import Any

# Import and re-export submodules for proper module structure
from . import io
from . import layouts
from . import models
from . import palettes
from . import plotting
from . import transform

# Direct re-exports for backward compatibility
from .models import ColorBar, ColumnDataSource, HoverTool, LinearColorMapper
from .plotting import figure, output_file, save
from .io import curdoc, push_notebook, show
from .layouts import column, gridplot, row

# Catch-all for any missing bokeh attributes
def __getattr__(name: str) -> Any: ...
