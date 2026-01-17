# Progress Bar Protocol Design (Issue #587)

**Date**: 2026-01-17  
**Status**: Proposed  
**Type**: Architecture Decision Record  
**Related**: Issue #543 (hasattr elimination), Issue #587  

---

## Context

Issue #543 (hasattr elimination) identified 7 progress bar patterns using duck typing:

```python
# Current pattern
picard_range = tqdm(range(max_iter)) if verbose else range(max_iter)
if verbose and hasattr(picard_range, "set_postfix"):  # ❌ Duck typing
    picard_range.set_postfix(error=error)
```

These were initially documented as "acceptable backend compatibility", but an external expert review correctly identified this as **too permissive**. The real solution is architectural: use **Protocol pattern + Null Object pattern**.

## Problem Statement

### Current Issues

1. **Type Safety**: `Union[range, RichProgressBar]` - static analyzers cannot verify method calls
2. **hasattr Violations**: 7 patterns violate Issue #543 standards
3. **Separation of Concerns**: UI logic mixed into solver algorithms
4. **Testing**: Requires mocking tqdm/Rich behavior
5. **Extensibility**: Difficult to add WebSocket, Jupyter, or custom progress reporters

### Root Cause

**Duck typing (hasattr)** is used because `verbose=False` returns `range()` which has no UI methods. This forces runtime attribute checking at every usage site.

## Decision

Adopt **Protocol + Null Object + Adapter** pattern:

1. **ProgressTracker Protocol**: Define explicit contract
2. **RichProgressBar**: Adapter wrapping Rich's API
3. **NoOpProgressBar**: Null Object with same interface, zero overhead
4. **create_progress_bar()**: Factory ensuring type consistency

## Design

### 1. Protocol Definition

```python
from typing import Protocol, TypeVar, runtime_checkable, Iterator

T = TypeVar("T")

@runtime_checkable
class ProgressTracker(Protocol[T]):
    """
    Structural type for progress reporting.
    
    All implementations must provide:
    - __iter__: Iterate over sequence
    - update_metrics: Display metrics (replaces set_postfix)
    - log: Print messages (replaces write)
    
    Benefits:
    - Static type checking (Mypy verification)
    - No runtime hasattr checks needed
    - Clear contract for implementers
    """
    
    def __iter__(self) -> Iterator[T]: ...
    def update_metrics(self, **kwargs: Any) -> None: ...
    def log(self, message: str) -> None: ...
```

### 2. Rich Adapter

```python
class RichProgressBar:
    """
    Adapter Pattern: Wraps Rich's Progress API to match ProgressTracker.
    
    Responsibilities:
    - Translate update_metrics() to Rich's format
    - Handle log() safely (console.print above progress bar)
    - Manage Progress context lifecycle
    """
    
    def update_metrics(self, **kwargs: Any) -> None:
        self.postfix_data.update(kwargs)
        if self.progress and self.task_id is not None:
            metrics = ", ".join(
                f"{k}={v:.2e}" if isinstance(v, float) else f"{k}={v}"
                for k, v in self.postfix_data.items()
            )
            self.progress.update(
                self.task_id, 
                description=f"{self.desc} [{metrics}]"
            )
    
    def log(self, message: str) -> None:
        console.print(f"[blue]INFO[/]: {message}")
```

### 3. Null Object

```python
class NoOpProgressBar:
    """
    Null Object Pattern: Same interface, zero behavior.
    
    Key Properties:
    - Zero performance overhead (all methods are pass)
    - Eliminates conditional logic in solvers
    - Type-safe (satisfies ProgressTracker Protocol)
    
    Usage:
        progress = NoOpProgressBar(range(100))
        progress.update_metrics(error=1e-5)  # Does nothing
        progress.log("Done!")  # Does nothing
    """
    
    def __iter__(self) -> Iterator[Any]:
        yield from self.iterable  # Pass-through
    
    def update_metrics(self, **kwargs: Any) -> None:
        pass
    
    def log(self, message: str) -> None:
        pass
```

### 4. Factory Function

```python
def create_progress_bar(
    iterable: Any,
    *,
    verbose: bool = True,
    desc: str = "",
    total: int | None = None,
) -> ProgressTracker:
    """
    Factory ensuring consistent ProgressTracker return type.
    
    Polymorphism guarantees:
    - verbose=True → RichProgressBar (renders UI)
    - verbose=False → NoOpProgressBar (silent)
    - Both satisfy ProgressTracker Protocol
    - Solver code works identically with either
    
    Example:
        progress = create_progress_bar(range(100), verbose=True)
        for i in progress:
            progress.update_metrics(error=compute_error())
            if converged:
                progress.log("Converged!")
                break
    """
    return (
        RichProgressBar(iterable, desc=desc, total=total)
        if verbose
        else NoOpProgressBar(iterable, desc=desc, total=total)
    )
```

## Migration Path

### Phase 1: Extend API (Non-Breaking)

Add new infrastructure without breaking existing code:

```python
# mfg_pde/utils/progress.py
+ class ProgressTracker(Protocol[T]): ...
+ class NoOpProgressBar: ...
+ def create_progress_bar(...) -> ProgressTracker: ...

# Existing RichProgressBar gains new methods:
+ def update_metrics(self, **kwargs): ...  # New
+ def log(self, message: str): ...  # New
  def set_postfix(self, **kwargs): ...  # Deprecated but still works
```

### Phase 2: Migrate Solvers

**7 Files to Update**:
1. `fixed_point_iterator.py` (2 patterns)
2. `fictitious_play.py` (2 patterns)
3. `block_iterators.py` (2 patterns)
4. `hjb_gfdm.py` (1 pattern)

**Before**:
```python
picard_range = tqdm(range(max_iter)) if verbose else range(max_iter)

for i in picard_range:
    if verbose and hasattr(picard_range, "set_postfix"):  # ❌
        picard_range.set_postfix(error=error)
    if converged and verbose and hasattr(picard_range, "write"):  # ❌
        picard_range.write("Converged!")
```

**After**:
```python
from mfg_pde.utils.progress import create_progress_bar

progress = create_progress_bar(range(max_iter), verbose=verbose, desc="Picard")

for i in progress:
    progress.update_metrics(error=error)  # ✅ Always works
    if converged:
        progress.log("Converged!")  # ✅ Always works
```

### Phase 3: Cleanup

- Remove #543 "acceptable" tags from progress bar patterns
- Update CLAUDE.md with new pattern
- Add deprecation warnings to `set_postfix()` and legacy methods
- Document in DEPRECATION_MODERNIZATION_GUIDE.md

## Consequences

### Positive

✅ **Zero hasattr checks** in solver code  
✅ **Type safety**: Mypy verifies all `ProgressTracker` calls  
✅ **Performance**: `NoOpProgressBar` is zero-overhead pass-through  
✅ **Testability**: `NoOpProgressBar` is built-in test double  
✅ **Extensibility**: Easy to add `WebSocketProgressBar`, `JupyterProgressBar`, etc.  
✅ **Separation of concerns**: UI logic isolated from algorithms  

### Neutral

⚠️ **New abstraction**: Developers must learn `ProgressTracker` Protocol  
⚠️ **Migration effort**: 7 files need updating (~3 hours work)  

### Negative

❌ **None identified** - This is a strict improvement over current design

## Alternatives Considered

### Alternative 1: Keep hasattr (Status Quo)

**Rejected**: Violates Issue #543 standards, poor type safety, mixes concerns

### Alternative 2: Always return RichProgressBar with disable=True

**Rejected**: Performance overhead (object creation + method calls even when disabled)

### Alternative 3: Use ABC instead of Protocol

**Rejected**: Requires inheritance, less flexible than structural typing

## Implementation

See **Issue #587** for detailed implementation plan.

**Estimated Effort**: 5-7 hours total
- Phase 1 (API): 2-3 hours
- Phase 2 (Migration): 2-3 hours
- Phase 3 (Docs): 1 hour

## Success Metrics

- ✅ Zero hasattr checks in progress bar code
- ✅ All solver tests passing
- ✅ Mypy verification of ProgressTracker usage
- ✅ Performance: NoOpProgressBar = plain range() speed
- ✅ Backward compatibility: Old code still works with deprecation warnings

## References

- **Issue #543**: hasattr elimination project
- **Issue #587**: Progress bar Protocol refactoring
- **PEP 544**: Protocols (Structural Subtyping)
- **Design Patterns**: Null Object, Adapter, Factory
- **External Expert Review**: Recommended this approach over "acceptable" categorization

---

**Last Updated**: 2026-01-17  
**Decision**: Approved for implementation  
**Tracking**: Issue #587
