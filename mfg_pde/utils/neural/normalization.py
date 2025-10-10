"""
Normalization Type Selection for Neural Networks.

Provides a clean enum-based API for selecting normalization types
in neural network architectures instead of using multiple boolean flags.
"""

from enum import Enum


class NormalizationType(str, Enum):
    """
    Normalization type selection for neural network layers.

    This enum provides a clean API for selecting which normalization
    technique to apply in neural network architectures.

    Attributes:
        NONE: No normalization
        BATCH: Batch normalization (normalizes across batch dimension)
        LAYER: Layer normalization (normalizes across feature dimension)

    Example:
        >>> from mfg_pde.utils.neural.normalization import NormalizationType
        >>> config = DeepONetConfig(normalization=NormalizationType.LAYER)
        >>> # Instead of: use_batch_norm=False, use_layer_norm=True
    """

    NONE = "none"  # No normalization
    BATCH = "batch"  # Batch normalization
    LAYER = "layer"  # Layer normalization

    @property
    def is_none(self) -> bool:
        """Check if no normalization is used."""
        return self == NormalizationType.NONE

    @property
    def is_batch(self) -> bool:
        """Check if batch normalization is used."""
        return self == NormalizationType.BATCH

    @property
    def is_layer(self) -> bool:
        """Check if layer normalization is used."""
        return self == NormalizationType.LAYER

    @property
    def requires_pytorch(self) -> bool:
        """Check if normalization requires PyTorch."""
        # All normalization types require PyTorch
        return self != NormalizationType.NONE

    def get_pytorch_module_name(self) -> str | None:
        """
        Get the name of the PyTorch module for this normalization type.

        Returns:
            Module name (e.g., "BatchNorm1d", "LayerNorm") or None
        """
        if self == NormalizationType.BATCH:
            return "BatchNorm1d"
        elif self == NormalizationType.LAYER:
            return "LayerNorm"
        return None
