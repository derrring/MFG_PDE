# Attrs stub for MFG_PDE strategic typing
# Focus on dataclass-style functionality used in configuration

from collections.abc import Callable
from typing import Any

# Core attrs decorators
def s(cls: type | None = None, **kwargs: Any) -> Any: ...  # @attr.s decorator
def attrs(cls: type | None = None, **kwargs: Any) -> Any: ...  # @attrs decorator

# Attribute definition
def ib(**kwargs: Any) -> Any: ...  # attr.ib() field definition
def attrib(**kwargs: Any) -> Any: ...  # attrs.attrib() field definition

# Validation functions
def instance_of(type_: type) -> Callable[[Any, Any, Any], None]: ...
def in_(collection: Any) -> Callable[[Any, Any, Any], None]: ...

# Conversion functions
def converters() -> Any: ...

# Utility functions
def asdict(inst: Any, **kwargs: Any) -> dict[str, Any]: ...
def astuple(inst: Any, **kwargs: Any) -> tuple[Any, ...]: ...

# Catch-all for any missing attrs attributes
def __getattr__(name: str) -> Any: ...
