# Bokeh palettes stub for MFG_PDE strategic typing

from typing import Any

# Palettes used in MFG_PDE
Inferno256: list[str]
Plasma256: list[str]
Viridis256: list[str]

# Catch-all for any missing palettes
def __getattr__(name: str) -> Any: ...
