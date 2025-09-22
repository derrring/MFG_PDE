# Package-Wide Typing Modernization Plan

## 🎯 **Objective**
Systematically eliminate all Pylance errors across the MFG_PDE package to achieve consistent, modern Python 3.12+ typing.

## 📊 **Error Pattern Analysis**

Based on work across multiple modules, the recurring error patterns are:

### 1. **Pydantic Configuration Issues**
- ❌ `ConfigDict` constructor syntax problems
- ❌ `Field(default_factory=Class)` instead of callable factories
- ❌ Missing constructor arguments in factory methods
- ✅ **Solution**: Use dict syntax for model_config, lambda factories, explicit arguments

### 2. **JAX/NumPy Compatibility Issues**
- ❌ `.at` attribute access on numpy arrays
- ❌ Type unions not handling both JAX and numpy arrays
- ❌ Runtime detection not understood by static analysis
- ✅ **Solution**: Use `getattr()`, proper type unions, runtime dispatching

### 3. **Type Alias and Forward Reference Issues**
- ❌ Complex types not using TypeAlias
- ❌ Runtime variables used in type expressions
- ❌ Missing TYPE_CHECKING imports
- ✅ **Solution**: Centralized TypeAlias definitions, quoted forward references

### 4. **Import Resolution Issues**
- ❌ Conditional imports not properly handled
- ❌ Optional dependencies causing import errors
- ❌ Missing fallback implementations
- ✅ **Solution**: Proper TYPE_CHECKING blocks, fallback imports

## 🛠️ **Systematic Fix Strategy**

### **Phase 1: Core Type Infrastructure** ✅ COMPLETED
- [x] `mfg_pde/types/internal.py` - Central TypeAlias definitions
- [x] JAXArray type handling both JAX and numpy arrays
- [x] Common solver return types

### **Phase 2: Configuration System** 🔄 IN PROGRESS
- [x] `mfg_pde/config/pydantic_config.py` - Pydantic v2 syntax fixes
- [ ] `mfg_pde/config/omegaconf_manager.py` - OmegaConf typing
- [ ] `mfg_pde/config/solver_config.py` - Legacy config compatibility

### **Phase 3: JAX Acceleration** ✅ COMPLETED
- [x] `mfg_pde/accelerated/jax_mfg_solver.py` - JAX solver typing
- [x] `mfg_pde/accelerated/jax_utils.py` - JAX utilities typing
- [x] Runtime JAX/numpy compatibility

### **Phase 4: Core Mathematical Components** 📋 PENDING
- [ ] `mfg_pde/core/` - Core problem definitions
- [ ] `mfg_pde/alg/` - Algorithm implementations
- [ ] `mfg_pde/geometry/` - Geometry and mesh classes

### **Phase 5: Utilities and Visualization** 📋 PENDING
- [ ] `mfg_pde/utils/` - Utility functions
- [ ] `mfg_pde/visualization/` - Plotting and analysis
- [ ] `mfg_pde/workflow/` - Workflow management

## 🔧 **Standard Fix Patterns**

### **Pattern 1: Pydantic Classes**
```python
# ❌ Before
model_config = ConfigDict(env_prefix="PREFIX_", validate_assignment=True)
field: SomeClass = Field(default_factory=SomeClass)

# ✅ After
model_config = {"env_prefix": "PREFIX_", "validate_assignment": True}
field: SomeClass = Field(default_factory=lambda: SomeClass())
```

### **Pattern 2: JAX/NumPy Compatibility**
```python
# ❌ Before
if hasattr(array, 'at'):
    array = array.at[0].set(value)

# ✅ After
if HAS_JAX and hasattr(array, 'at'):
    array = getattr(array, 'at')[0].set(value)
else:
    array = array.copy()
    array[0] = value
```

### **Pattern 3: Type Aliases**
```python
# ❌ Before
def func(param: jnp.ndarray) -> jnp.ndarray:

# ✅ After
def func(param: "JAXArray") -> "JAXArray":
```

### **Pattern 4: Import Safety**
```python
# ❌ Before
from optional_package import SomeClass

# ✅ After
if TYPE_CHECKING:
    from optional_package import SomeClass
else:
    try:
        from optional_package import SomeClass
    except ImportError:
        SomeClass = Any
```

## 📝 **Implementation Steps**

### **Step 1: Finish pydantic_config.py** 🔄 CURRENT
1. Fix remaining ConfigDict issues
2. Ensure all default_factory uses lambdas
3. Add missing constructor arguments
4. Test all factory methods

### **Step 2: Apply to Other Config Modules**
1. omegaconf_manager.py - Fix DictConfig typing
2. solver_config.py - Legacy compatibility
3. array_validation.py - Array type validation

### **Step 3: Core Module Modernization**
1. Scan all .py files for common error patterns
2. Apply standard fix patterns consistently
3. Test imports and basic functionality
4. Validate with pylance/mypy

### **Step 4: Package-Wide Validation**
1. Run type checking on entire package
2. Fix any remaining edge cases
3. Update documentation
4. Create typing best practices guide

## 🎯 **Success Criteria**

- [ ] Zero Pylance errors across entire package
- [ ] All modules import without errors
- [ ] Type checking passes with strict settings
- [ ] Runtime functionality preserved
- [ ] Documentation updated

## 📈 **Progress Tracking**

- ✅ JAX acceleration modules (100% complete)
- 🔄 Configuration modules (60% complete)
- 📋 Core modules (0% complete)
- 📋 Utility modules (0% complete)
- 📋 Visualization modules (0% complete)

**Total Progress: ~20% complete**

## 🚀 **Next Actions**

1. **IMMEDIATE**: Complete pydantic_config.py fixes
2. **SHORT-TERM**: Apply patterns to config/ directory
3. **MEDIUM-TERM**: Modernize core/ and alg/ directories
4. **LONG-TERM**: Complete utils/ and visualization/ directories

---

**Created**: 2025-01-23
**Last Updated**: 2025-01-23
**Status**: Active Implementation
