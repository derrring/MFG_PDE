"""
Internal Implementation Details

⚠️  MAINTAINERS ONLY - DO NOT USE IN USER CODE

This module contains complex internal types and implementations
that are subject to change without notice. These are used internally
by the library but are not part of the public API.

If you find yourself importing from this module, you probably
want to use the hooks system instead:

    from mfg_pde.hooks import SolverHooks

    class MyCustomHook(SolverHooks):
        def on_iteration_end(self, state):
            # Your custom logic here
            pass
"""

from __future__ import annotations

# Explicitly empty - internal modules should be imported directly
# This prevents accidental import of internal APIs
from typing import List

__all__: list[str] = []
