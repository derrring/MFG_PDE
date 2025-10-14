#!/usr/bin/env python3
"""
Unit tests for mfg_pde/utils/neural/normalization.py

Tests NormalizationType enum including:
- Enum values and string representation
- Normalization type checks (is_none, is_batch, is_layer)
- PyTorch dependency checking (requires_pytorch, get_pytorch_module_name)
- Enum behavior (comparison, iteration, membership)
"""

import pytest

from mfg_pde.utils.neural.normalization import NormalizationType

# ===================================================================
# Test NormalizationType Enum Values
# ===================================================================


@pytest.mark.unit
def test_normalization_type_values():
    """Test NormalizationType enum has expected values."""
    assert NormalizationType.NONE.value == "none"
    assert NormalizationType.BATCH.value == "batch"
    assert NormalizationType.LAYER.value == "layer"


@pytest.mark.unit
def test_normalization_type_string_representation():
    """Test NormalizationType string representation."""
    # str() returns full name like "NormalizationType.NONE"
    assert "NONE" in str(NormalizationType.NONE)
    assert "BATCH" in str(NormalizationType.BATCH)
    assert "LAYER" in str(NormalizationType.LAYER)


@pytest.mark.unit
def test_normalization_type_from_string():
    """Test creating NormalizationType from string."""
    assert NormalizationType("none") == NormalizationType.NONE
    assert NormalizationType("batch") == NormalizationType.BATCH
    assert NormalizationType("layer") == NormalizationType.LAYER


@pytest.mark.unit
def test_normalization_type_invalid_string():
    """Test creating NormalizationType from invalid string raises error."""
    with pytest.raises(ValueError):
        NormalizationType("instance")
    with pytest.raises(ValueError):
        NormalizationType("invalid")


# ===================================================================
# Test is_none Property
# ===================================================================


@pytest.mark.unit
def test_is_none_true():
    """Test is_none property returns True for NONE type."""
    assert NormalizationType.NONE.is_none is True


@pytest.mark.unit
def test_is_none_false():
    """Test is_none property returns False for non-NONE types."""
    assert NormalizationType.BATCH.is_none is False
    assert NormalizationType.LAYER.is_none is False


# ===================================================================
# Test is_batch Property
# ===================================================================


@pytest.mark.unit
def test_is_batch_true():
    """Test is_batch property returns True for BATCH type."""
    assert NormalizationType.BATCH.is_batch is True


@pytest.mark.unit
def test_is_batch_false():
    """Test is_batch property returns False for non-BATCH types."""
    assert NormalizationType.NONE.is_batch is False
    assert NormalizationType.LAYER.is_batch is False


# ===================================================================
# Test is_layer Property
# ===================================================================


@pytest.mark.unit
def test_is_layer_true():
    """Test is_layer property returns True for LAYER type."""
    assert NormalizationType.LAYER.is_layer is True


@pytest.mark.unit
def test_is_layer_false():
    """Test is_layer property returns False for non-LAYER types."""
    assert NormalizationType.NONE.is_layer is False
    assert NormalizationType.BATCH.is_layer is False


# ===================================================================
# Test requires_pytorch Property
# ===================================================================


@pytest.mark.unit
def test_requires_pytorch_none():
    """Test NONE type does not require PyTorch."""
    assert NormalizationType.NONE.requires_pytorch is False


@pytest.mark.unit
def test_requires_pytorch_batch():
    """Test BATCH type requires PyTorch."""
    assert NormalizationType.BATCH.requires_pytorch is True


@pytest.mark.unit
def test_requires_pytorch_layer():
    """Test LAYER type requires PyTorch."""
    assert NormalizationType.LAYER.requires_pytorch is True


# ===================================================================
# Test get_pytorch_module_name Method
# ===================================================================


@pytest.mark.unit
def test_get_pytorch_module_name_none():
    """Test get_pytorch_module_name for NONE returns None."""
    assert NormalizationType.NONE.get_pytorch_module_name() is None


@pytest.mark.unit
def test_get_pytorch_module_name_batch():
    """Test get_pytorch_module_name for BATCH returns 'BatchNorm1d'."""
    assert NormalizationType.BATCH.get_pytorch_module_name() == "BatchNorm1d"


@pytest.mark.unit
def test_get_pytorch_module_name_layer():
    """Test get_pytorch_module_name for LAYER returns 'LayerNorm'."""
    assert NormalizationType.LAYER.get_pytorch_module_name() == "LayerNorm"


# ===================================================================
# Test Enum Behavior
# ===================================================================


@pytest.mark.unit
def test_normalization_type_equality():
    """Test NormalizationType equality comparison."""
    assert NormalizationType.NONE == NormalizationType.NONE
    assert NormalizationType.BATCH == NormalizationType.BATCH
    assert NormalizationType.LAYER == NormalizationType.LAYER

    assert NormalizationType.NONE != NormalizationType.BATCH
    assert NormalizationType.BATCH != NormalizationType.LAYER


@pytest.mark.unit
def test_normalization_type_identity():
    """Test NormalizationType identity (is operator)."""
    none1 = NormalizationType.NONE
    none2 = NormalizationType.NONE
    assert none1 is none2  # Enum members are singletons


@pytest.mark.unit
def test_normalization_type_membership():
    """Test NormalizationType membership checks."""
    types = [NormalizationType.NONE, NormalizationType.BATCH]
    assert NormalizationType.NONE in types
    assert NormalizationType.LAYER not in types


@pytest.mark.unit
def test_normalization_type_iteration():
    """Test NormalizationType iteration."""
    types = list(NormalizationType)
    assert len(types) == 3
    assert NormalizationType.NONE in types
    assert NormalizationType.BATCH in types
    assert NormalizationType.LAYER in types


@pytest.mark.unit
def test_normalization_type_count():
    """Test NormalizationType has exactly 3 members."""
    assert len(list(NormalizationType)) == 3


# ===================================================================
# Test String Enum Behavior
# ===================================================================


@pytest.mark.unit
def test_normalization_type_str_enum_comparison():
    """Test NormalizationType can be compared with strings."""
    # Since NormalizationType inherits from str
    assert NormalizationType.NONE == "none"
    assert NormalizationType.BATCH == "batch"
    assert NormalizationType.LAYER == "layer"


@pytest.mark.unit
def test_normalization_type_string_operations():
    """Test NormalizationType supports string operations."""
    # Can use in string contexts
    assert NormalizationType.NONE.upper() == "NONE"
    assert NormalizationType.BATCH.startswith("b")
    assert NormalizationType.LAYER.endswith("r")


# ===================================================================
# Test Property Mutual Exclusivity
# ===================================================================


@pytest.mark.unit
def test_type_properties_mutually_exclusive():
    """Test type properties are mutually exclusive."""
    for norm_type in NormalizationType:
        # Each type has exactly one True property
        properties = [norm_type.is_none, norm_type.is_batch, norm_type.is_layer]
        assert sum(properties) == 1, f"{norm_type} has multiple True properties"


@pytest.mark.unit
def test_none_properties():
    """Test NONE type has correct property values."""
    norm_type = NormalizationType.NONE
    assert norm_type.is_none is True
    assert norm_type.is_batch is False
    assert norm_type.is_layer is False
    assert norm_type.requires_pytorch is False
    assert norm_type.get_pytorch_module_name() is None


@pytest.mark.unit
def test_batch_properties():
    """Test BATCH type has correct property values."""
    norm_type = NormalizationType.BATCH
    assert norm_type.is_none is False
    assert norm_type.is_batch is True
    assert norm_type.is_layer is False
    assert norm_type.requires_pytorch is True
    assert norm_type.get_pytorch_module_name() == "BatchNorm1d"


@pytest.mark.unit
def test_layer_properties():
    """Test LAYER type has correct property values."""
    norm_type = NormalizationType.LAYER
    assert norm_type.is_none is False
    assert norm_type.is_batch is False
    assert norm_type.is_layer is True
    assert norm_type.requires_pytorch is True
    assert norm_type.get_pytorch_module_name() == "LayerNorm"


# ===================================================================
# Test Usage Patterns
# ===================================================================


@pytest.mark.unit
def test_normalization_selection_pattern():
    """Test typical normalization selection pattern."""

    def select_normalization(norm_type: NormalizationType) -> str:
        if norm_type.is_none:
            return "No normalization"
        elif norm_type.is_batch:
            return "Using batch normalization"
        elif norm_type.is_layer:
            return "Using layer normalization"
        return "Unknown"

    assert select_normalization(NormalizationType.NONE) == "No normalization"
    assert select_normalization(NormalizationType.BATCH) == "Using batch normalization"
    assert select_normalization(NormalizationType.LAYER) == "Using layer normalization"


@pytest.mark.unit
def test_pytorch_module_selection_pattern():
    """Test typical PyTorch module selection pattern."""

    def get_module_name(norm_type: NormalizationType) -> str | None:
        if norm_type.requires_pytorch:
            return norm_type.get_pytorch_module_name()
        return None

    assert get_module_name(NormalizationType.NONE) is None
    assert get_module_name(NormalizationType.BATCH) == "BatchNorm1d"
    assert get_module_name(NormalizationType.LAYER) == "LayerNorm"


@pytest.mark.unit
def test_config_usage_pattern():
    """Test typical configuration usage pattern."""

    class MockConfig:
        def __init__(self, normalization: NormalizationType = NormalizationType.NONE):
            self.normalization = normalization

    config1 = MockConfig(normalization=NormalizationType.NONE)
    config2 = MockConfig(normalization=NormalizationType.BATCH)
    config3 = MockConfig(normalization=NormalizationType.LAYER)

    assert config1.normalization.is_none
    assert config2.normalization.is_batch
    assert config3.normalization.is_layer


@pytest.mark.unit
def test_conditional_import_pattern():
    """Test typical conditional import pattern based on normalization type."""

    def should_import_pytorch(norm_type: NormalizationType) -> bool:
        return norm_type.requires_pytorch

    assert should_import_pytorch(NormalizationType.NONE) is False
    assert should_import_pytorch(NormalizationType.BATCH) is True
    assert should_import_pytorch(NormalizationType.LAYER) is True


# ===================================================================
# Test Type Annotations
# ===================================================================


@pytest.mark.unit
def test_normalization_type_annotation():
    """Test NormalizationType works in type annotations."""

    def process_with_normalization(norm_type: NormalizationType) -> str:
        return norm_type.value

    assert process_with_normalization(NormalizationType.NONE) == "none"
    assert process_with_normalization(NormalizationType.BATCH) == "batch"
    assert process_with_normalization(NormalizationType.LAYER) == "layer"


# ===================================================================
# Test Module Exports
# ===================================================================


@pytest.mark.unit
def test_module_exports():
    """Test NormalizationType is importable."""
    from mfg_pde.utils.neural import normalization

    assert hasattr(normalization, "NormalizationType")
    assert normalization.NormalizationType == NormalizationType


@pytest.mark.unit
def test_module_docstring():
    """Test module has comprehensive docstring."""
    from mfg_pde.utils.neural import normalization

    assert normalization.__doc__ is not None
    assert "Normalization Type Selection" in normalization.__doc__


# ===================================================================
# Test PyTorch Module Names
# ===================================================================


@pytest.mark.unit
def test_all_pytorch_types_have_module_names():
    """Test all types requiring PyTorch have module names."""
    for norm_type in NormalizationType:
        if norm_type.requires_pytorch:
            module_name = norm_type.get_pytorch_module_name()
            assert module_name is not None
            assert isinstance(module_name, str)
            assert len(module_name) > 0


@pytest.mark.unit
def test_pytorch_module_name_format():
    """Test PyTorch module names follow expected format."""
    # Module names should be CamelCase and end with "Norm" or contain "Norm"
    for norm_type in NormalizationType:
        module_name = norm_type.get_pytorch_module_name()
        if module_name is not None:
            assert "Norm" in module_name
            assert module_name[0].isupper()  # CamelCase
