#!/usr/bin/env python3
"""
Test SKIPPED - Particle-Collocation Removed

NOTE: This test has been skipped because it depends on ParticleCollocationSolver,
which has been moved to the mfg-research repository as part of architectural separation.

For particle-collocation tests, see: mfg-research/algorithms/particle_collocation/tests/
Migration documentation: MIGRATION_PARTICLE_COLLOCATION.md
"""

import pytest


@pytest.mark.skip(
    reason="Test uses ParticleCollocationSolver which has been moved to mfg-research. "
    "See MIGRATION_PARTICLE_COLLOCATION.md"
)
def test_particle_collocation_dependency_removed():
    """Placeholder indicating this test depends on moved functionality."""
