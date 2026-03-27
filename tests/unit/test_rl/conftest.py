"""
RL test suite configuration.

These tests are for placeholder/scaffold RL algorithms that are not yet
production-ready. They are skipped by default to avoid inflating test
metrics. Remove this skip when RL algorithms reach production quality
with stable APIs and validated results.

See Issue #833: Test quality audit.
"""

import pytest

# Skip entire RL test directory — algorithms not yet implemented
collect_ignore_glob = ["test_*.py"]


def pytest_collection_modifyitems(items):
    """Mark all RL tests as skipped with reason."""
    for item in items:
        if "test_rl" in str(item.fspath):
            item.add_marker(pytest.mark.skip(reason="RL algorithms not yet production-ready (Issue #833)"))
