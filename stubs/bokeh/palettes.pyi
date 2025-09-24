# Bokeh palettes stub for MFG_PDE strategic typing

from typing import Any, List

# Palettes used in MFG_PDE
Inferno256: List[str]
Plasma256: List[str]
Viridis256: List[str]

# Catch-all for any missing palettes
def __getattr__(name: str) -> Any: ...
