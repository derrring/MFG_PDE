# MFG_PDE Research Investigations

This directory contains research-grade investigation scripts and exploratory tests that are **not part of the CI test suite**. These artifacts document historical bug investigations, mathematical validations, and exploratory research.

## Directory Structure

### `bug_investigations/`
Historical bug diagnosis and resolution scripts.
- Not regression tests - one-time investigation artifacts
- Preserved for documentation and future reference
- Scripts may have dependencies on specific problem configurations

### `boundary_conditions/`
Research-grade boundary condition validation studies.
- Comparative studies of different BC implementations
- SVD-based no-flux condition experiments
- Scaling and convergence analyses

### `svd_implementation/`
SVD-based solver implementation explorations.
- Original parameter space investigations
- Second-order accuracy studies
- Implementation verification scripts

### `ghost_particles/`
Particle method investigations for boundary handling.
- Ghost particle placement strategies
- Boundary flux computation experiments

### `mathematical/`
Mathematical property verification and validation.
- Analytical solution comparisons
- Theoretical property checks
- Numerical method validation

### `property_based/`
Property-based testing experiments using Hypothesis or similar frameworks.
- Generative test exploration
- Edge case discovery

## Usage

These scripts are intended for **manual execution** during research and development:

```bash
# Run specific investigation
python investigations/bug_investigations/diagnose_time_index_bug.py

# Run investigation tests (not in CI)
pytest investigations/ -v

# Run specific category
pytest investigations/boundary_conditions/ -v
```

## Not for CI

**Important**: These investigations are **excluded from CI** to keep the test suite focused on stable core functionality. They represent:
- Completed bug investigations (resolved)
- One-time mathematical validations
- Research explorations
- Historical artifacts

For regression testing of stable features, add tests to `tests/unit/` or `tests/integration/`.

## Adding New Investigations

Place new investigations here when:
- Debugging a specific issue
- Exploring mathematical properties
- Validating new numerical methods
- Conducting comparative studies

Keep investigations here for documentation, but migrate essential validation logic to `tests/` once the feature is stable.

## Maintenance

- These scripts may depend on older APIs
- Not guaranteed to run with current codebase
- Preserved for historical context and reference
- Can be removed if disk space becomes an issue
