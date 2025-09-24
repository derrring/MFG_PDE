# Minimal attr stub for MFG_PDE strategic typing
# Legacy compatibility for attrs library

# Import everything from attrs for compatibility
from attrs import *  # noqa: F403

# Catch-all for any missing attr attributes
def __getattr__(name: str): ...
