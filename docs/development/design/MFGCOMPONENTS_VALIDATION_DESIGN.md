# MFGComponents Validation Design

**Date**: 2025-11-03
**Purpose**: Design validation logic for MFGComponents consistency
**Status**: Design specification (implementation pending)

---

## Motivation

With MFGComponents now supporting 6+ different MFG formulations (standard, network, variational, neural, RL, etc.), validation is needed to:

1. **Detect inconsistencies**: Neural architecture provided but no Hamiltonian
2. **Warn about incomplete configurations**: RL reward but no action space
3. **Prevent silent failures**: Conflicting formulations specified
4. **Guide users**: Helpful error messages about what's missing

---

## Design Principles

### **1. Non-Breaking**
All validation is **warnings**, not errors. Users can ignore if they know what they're doing.

### **2. Formulation-Aware**
Different formulations have different requirements:
- Standard HJB-FP needs Hamiltonian or potential
- Neural MFG needs architecture + Hamiltonian
- RL MFG needs reward + action space
- Network MFG needs network geometry

### **3. Helpful Messages**
Tell users **what's missing** and **why it matters**:
```python
"Neural MFG: neural_architecture provided but hamiltonian_func is None.
 Neural solvers need a Hamiltonian to learn the value function."
```

### **4. Optional Invocation**
Users choose when to validate:
```python
components = MFGComponents(...)
warnings = components.validate()  # Returns list of warning strings
if warnings:
    for w in warnings:
        print(f"Warning: {w}")
```

---

## Validation Categories

### **Category 1: Formulation Consistency**

Check that related components are provided together.

#### **Standard HJB-FP**
```python
def _validate_standard_mfg(self) -> list[str]:
    warnings = []

    # Need at least one of: Hamiltonian, potential, or coupling
    has_dynamics = any([
        self.hamiltonian_func is not None,
        self.potential_func is not None,
        self.coupling_func is not None
    ])

    if not has_dynamics and self.network_geometry is None:
        warnings.append(
            "Standard MFG: No dynamics specified. "
            "Provide hamiltonian_func, potential_func, or coupling_func."
        )

    # If Hamiltonian provided, suggest derivatives for efficiency
    if self.hamiltonian_func is not None:
        if self.hamiltonian_dm_func is None:
            warnings.append(
                "Performance: hamiltonian_func provided but hamiltonian_dm_func is None. "
                "Providing analytical derivative improves efficiency."
            )

    return warnings
```

#### **Neural Network MFG**
```python
def _validate_neural_mfg(self) -> list[str]:
    warnings = []

    # Neural architecture needs underlying problem definition
    if self.neural_architecture is not None:
        if self.hamiltonian_func is None:
            warnings.append(
                "Neural MFG: neural_architecture provided but hamiltonian_func is None. "
                "Neural solvers need a Hamiltonian to define the PDE."
            )

        # Check loss weights are sensible
        if self.loss_weights is not None:
            if any(w < 0 for w in self.loss_weights.values()):
                warnings.append(
                    "Neural MFG: loss_weights contains negative values. "
                    "Weights should be non-negative."
                )

    # Specific network configs need general architecture
    specific_configs = [
        self.value_network_config,
        self.policy_network_config,
        self.density_network_config
    ]

    if any(c is not None for c in specific_configs):
        if self.neural_architecture is None:
            warnings.append(
                "Neural MFG: Specific network config provided but neural_architecture is None. "
                "Provide general architecture or use specific configs alone."
            )

    return warnings
```

#### **Reinforcement Learning MFG**
```python
def _validate_rl_mfg(self) -> list[str]:
    warnings = []

    # RL needs reward and action space
    if self.reward_func is not None:
        if self.action_space_bounds is None:
            warnings.append(
                "RL MFG: reward_func provided but action_space_bounds is None. "
                "RL requires action space definition."
            )

    # Action constraints need action space
    if self.action_constraints is not None:
        if self.action_space_bounds is None:
            warnings.append(
                "RL MFG: action_constraints provided but action_space_bounds is None. "
                "Constraints require action space definition."
            )

    # Check population coupling strength is reasonable
    if self.population_coupling_strength < 0:
        warnings.append(
            "RL MFG: population_coupling_strength is negative. "
            "Coupling strength should be non-negative."
        )

    return warnings
```

#### **Network/Graph MFG**
```python
def _validate_network_mfg(self) -> list[str]:
    warnings = []

    # Network geometry needs interaction or cost functions
    if self.network_geometry is not None:
        has_interactions = any([
            self.node_interaction_func is not None,
            self.edge_interaction_func is not None,
            self.edge_cost_func is not None,
            self.trajectory_cost_func is not None
        ])

        if not has_interactions:
            warnings.append(
                "Network MFG: network_geometry provided but no interaction functions. "
                "Specify node_interaction_func, edge_interaction_func, or edge_cost_func."
            )

    return warnings
```

#### **Variational/Lagrangian MFG**
```python
def _validate_variational_mfg(self) -> list[str]:
    warnings = []

    # Lagrangian needs derivatives for optimization
    if self.lagrangian_func is not None:
        missing_derivatives = []
        if self.lagrangian_dx_func is None:
            missing_derivatives.append("lagrangian_dx_func")
        if self.lagrangian_dv_func is None:
            missing_derivatives.append("lagrangian_dv_func")

        if missing_derivatives:
            warnings.append(
                f"Variational MFG: lagrangian_func provided but missing derivatives: "
                f"{', '.join(missing_derivatives)}. "
                f"Analytical derivatives improve optimization."
            )

    # Terminal cost derivative for boundary conditions
    if self.terminal_cost_func is not None:
        if self.terminal_cost_dx_func is None:
            warnings.append(
                "Variational MFG: terminal_cost_func provided but terminal_cost_dx_func is None. "
                "Derivative needed for terminal conditions."
            )

    return warnings
```

#### **Implicit Geometry**
```python
def _validate_implicit_geometry(self) -> list[str]:
    warnings = []

    # Level set needs signed distance for efficiency
    if self.level_set_func is not None:
        if self.signed_distance_func is None:
            warnings.append(
                "Implicit Geometry: level_set_func provided but signed_distance_func is None. "
                "SDF improves distance queries and normal computations."
            )

    # Obstacle function needs reasonable penalty
    if self.obstacle_func is not None:
        if self.obstacle_penalty <= 0:
            warnings.append(
                "Implicit Geometry: obstacle_penalty must be positive. "
                f"Current value: {self.obstacle_penalty}"
            )

    # Manifold projection needs tangent space for consistency
    if self.manifold_projection is not None:
        if self.tangent_space_basis is None:
            warnings.append(
                "Implicit Geometry: manifold_projection provided but tangent_space_basis is None. "
                "Tangent space needed for velocity projections."
            )

    return warnings
```

#### **Adaptive Mesh Refinement**
```python
def _validate_amr(self) -> list[str]:
    warnings = []

    # Refinement indicator needs thresholds
    if self.refinement_indicator is not None:
        if self.refinement_threshold <= self.coarsening_threshold:
            warnings.append(
                f"AMR: refinement_threshold ({self.refinement_threshold}) must be > "
                f"coarsening_threshold ({self.coarsening_threshold}). "
                f"Otherwise refinement/coarsening will conflict."
            )

    # Cell size constraints
    if self.min_cell_size is not None and self.max_cell_size is not None:
        if self.min_cell_size >= self.max_cell_size:
            warnings.append(
                f"AMR: min_cell_size ({self.min_cell_size}) must be < "
                f"max_cell_size ({self.max_cell_size})."
            )

    # Refinement level should be reasonable
    if self.max_refinement_level > 10:
        warnings.append(
            f"AMR: max_refinement_level ({self.max_refinement_level}) is very large. "
            f"Typical values are 3-7."
        )

    return warnings
```

#### **Multi-Population MFG**
```python
def _validate_multi_population(self) -> list[str]:
    warnings = []

    if self.num_populations > 1:
        # Need population-specific Hamiltonians
        if self.population_hamiltonians is not None:
            if len(self.population_hamiltonians) != self.num_populations:
                warnings.append(
                    f"Multi-Population: num_populations={self.num_populations} but "
                    f"population_hamiltonians has {len(self.population_hamiltonians)} entries. "
                    f"Counts must match."
                )

        # Need population-specific initial densities
        if self.population_initial_densities is not None:
            if len(self.population_initial_densities) != self.num_populations:
                warnings.append(
                    f"Multi-Population: num_populations={self.num_populations} but "
                    f"population_initial_densities has {len(self.population_initial_densities)} entries. "
                    f"Counts must match."
                )

        # Population weights should sum to reasonable value
        if self.population_weights is not None:
            if len(self.population_weights) != self.num_populations:
                warnings.append(
                    f"Multi-Population: num_populations={self.num_populations} but "
                    f"population_weights has {len(self.population_weights)} entries."
                )

            total_weight = sum(self.population_weights)
            if abs(total_weight - 1.0) > 1e-6:
                warnings.append(
                    f"Multi-Population: population_weights sum to {total_weight:.6f}, not 1.0. "
                    f"Consider normalizing."
                )

    return warnings
```

### **Category 2: Formulation Conflicts**

Check for mutually exclusive or conflicting configurations.

```python
def _validate_formulation_conflicts(self) -> list[str]:
    warnings = []

    # Count active formulations
    has_standard = self.hamiltonian_func is not None
    has_variational = self.lagrangian_func is not None
    has_network = self.network_geometry is not None
    has_neural = self.neural_architecture is not None
    has_rl = self.reward_func is not None

    active_formulations = sum([has_standard, has_variational, has_network])

    # Multiple formulations can coexist, but warn if unclear
    if active_formulations > 1:
        formulations = []
        if has_standard:
            formulations.append("Standard HJB-FP")
        if has_variational:
            formulations.append("Variational")
        if has_network:
            formulations.append("Network MFG")

        warnings.append(
            f"Multiple formulations detected: {', '.join(formulations)}. "
            f"Ensure solver supports hybrid formulations."
        )

    # Neural and RL are *methods* not formulations, can coexist with others
    # No conflict warnings needed

    return warnings
```

### **Category 3: Dimension Consistency**

Check that array dimensions match expected spatial dimension.

```python
def _validate_dimensions(self) -> list[str]:
    warnings = []

    # Correlation matrix should be square
    if self.correlation_matrix is not None:
        if self.correlation_matrix.ndim != 2:
            warnings.append(
                f"Stochastic: correlation_matrix must be 2D, got {self.correlation_matrix.ndim}D."
            )
        elif self.correlation_matrix.shape[0] != self.correlation_matrix.shape[1]:
            warnings.append(
                f"Stochastic: correlation_matrix must be square, "
                f"got shape {self.correlation_matrix.shape}."
            )

    # Action space bounds dimension consistency
    if self.action_space_bounds is not None:
        for i, (lower, upper) in enumerate(self.action_space_bounds):
            if lower >= upper:
                warnings.append(
                    f"RL MFG: action_space_bounds[{i}] has lower={lower} >= upper={upper}. "
                    f"Lower bound must be < upper bound."
                )

    return warnings
```

---

## Main Validation Method

```python
class MFGComponents:
    # ... all existing fields ...

    def validate(self, strict: bool = False) -> list[str]:
        """
        Validate component consistency and completeness.

        Parameters
        ----------
        strict : bool, default=False
            If True, raise ValueError on first warning.
            If False, return all warnings as list of strings.

        Returns
        -------
        warnings : list[str]
            List of warning messages. Empty if no issues found.

        Raises
        ------
        ValueError
            If strict=True and validation issues found.

        Examples
        --------
        >>> components = MFGComponents(
        ...     neural_architecture={'layers': [64, 64]},
        ...     # Missing hamiltonian_func
        ... )
        >>> warnings = components.validate()
        >>> if warnings:
        ...     for w in warnings:
        ...         print(f"Warning: {w}")

        >>> # Strict mode raises on issues
        >>> components.validate(strict=True)  # Raises ValueError
        """
        warnings = []

        # Run all validation checks
        warnings.extend(self._validate_standard_mfg())
        warnings.extend(self._validate_neural_mfg())
        warnings.extend(self._validate_rl_mfg())
        warnings.extend(self._validate_network_mfg())
        warnings.extend(self._validate_variational_mfg())
        warnings.extend(self._validate_implicit_geometry())
        warnings.extend(self._validate_amr())
        warnings.extend(self._validate_multi_population())
        warnings.extend(self._validate_formulation_conflicts())
        warnings.extend(self._validate_dimensions())

        if strict and warnings:
            error_msg = "MFGComponents validation failed:\n" + "\n".join(
                f"  - {w}" for w in warnings
            )
            raise ValueError(error_msg)

        return warnings
```

---

## Integration with MFGProblem

```python
class MFGProblem:
    def __init__(
        self,
        # ... existing parameters ...
        components: MFGComponents | None = None,
        validate_components: bool = True,
        **kwargs
    ):
        # ... existing init code ...

        # Validate components if provided
        if components is not None and validate_components:
            warnings = components.validate()
            if warnings:
                import warnings as warn_module
                for w in warnings:
                    warn_module.warn(f"MFGComponents: {w}", UserWarning, stacklevel=2)
```

---

## Testing Strategy

### **Unit Tests**

```python
# tests/unit/test_mfg_components_validation.py

def test_neural_mfg_missing_hamiltonian():
    """Neural architecture without Hamiltonian should warn."""
    components = MFGComponents(
        neural_architecture={'layers': [64, 64, 64]}
    )
    warnings = components.validate()
    assert any('hamiltonian_func' in w for w in warnings)


def test_rl_mfg_missing_action_space():
    """RL reward without action space should warn."""
    components = MFGComponents(
        reward_func=lambda s, a, m, t: -a**2
    )
    warnings = components.validate()
    assert any('action_space_bounds' in w for w in warnings)


def test_amr_invalid_thresholds():
    """AMR with refinement_threshold <= coarsening_threshold should warn."""
    components = MFGComponents(
        refinement_indicator=lambda u, m: abs(u),
        refinement_threshold=0.01,
        coarsening_threshold=0.1  # Wrong: larger than refinement
    )
    warnings = components.validate()
    assert any('threshold' in w.lower() for w in warnings)


def test_multi_population_mismatch():
    """Multi-population with mismatched array lengths should warn."""
    components = MFGComponents(
        num_populations=3,
        population_hamiltonians=[lambda: 0, lambda: 0]  # Only 2, need 3
    )
    warnings = components.validate()
    assert any('num_populations' in w for w in warnings)


def test_strict_mode_raises():
    """Strict mode should raise ValueError on issues."""
    components = MFGComponents(
        neural_architecture={'layers': [64, 64]}
    )
    with pytest.raises(ValueError, match="validation failed"):
        components.validate(strict=True)


def test_valid_components_no_warnings():
    """Properly configured components should pass validation."""
    components = MFGComponents(
        hamiltonian_func=lambda x, m, p, t: 0.5 * p**2 + m,
        initial_density_func=lambda x: np.exp(-x**2),
        boundary_conditions=BoundaryConditions()
    )
    warnings = components.validate()
    assert len(warnings) == 0
```

---

## Implementation Priority

### **Phase 1: Core Validation** (Immediate)
Implement validation for:
1. Standard HJB-FP consistency
2. Neural MFG requirements
3. RL MFG requirements
4. Dimension consistency

### **Phase 2: Advanced Validation** (Short-term)
Add validation for:
1. Network MFG
2. Variational MFG
3. Implicit geometry
4. Multi-population

### **Phase 3: Enhanced Validation** (Medium-term)
1. AMR configuration checks
2. Time-dependent domain validation
3. Formulation conflict detection
4. Cross-component consistency (e.g., boundary conditions match geometry)

---

## Benefits

1. **Catch user errors early**: Missing configurations detected at problem creation, not during solve
2. **Helpful guidance**: Clear messages about what's wrong and how to fix
3. **Non-breaking**: Optional validation, warnings not errors
4. **Documentation**: Validation logic documents expected relationships between components
5. **Testing aid**: Validation helps test suite verify problem configurations

---

## Backward Compatibility

**Impact**: ✅ 100% backward compatible

- Validation is **opt-in** via `components.validate()` call
- MFGProblem can enable by default but uses warnings (not errors)
- Existing code continues to work unchanged

---

## Summary

**Validation Design**: Category-based checks with helpful warnings

**Key Features**:
- ✅ Non-breaking (warnings, not errors)
- ✅ Formulation-aware (different requirements for different MFG types)
- ✅ Helpful messages (explain what's missing and why)
- ✅ Optional strict mode (raise on issues)

**Implementation**: Add `validate()` method and category validation helpers to MFGComponents

**Testing**: Comprehensive unit tests for each validation category

**Timeline**: Can implement Phase 1 (core validation) immediately

---

**Last Updated**: 2025-11-03
**Status**: Design complete, ready for implementation
**Next Steps**: Implement Phase 1 validation methods
