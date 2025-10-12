#!/usr/bin/env python3
"""
Particle-Collocation Integration Test - SKIPPED

NOTE: Particle-collocation methods have been moved to the mfg-research repository
as part of architectural separation between infrastructure and novel research algorithms.

For particle-collocation tests, see:
  mfg-research/algorithms/particle_collocation/tests/

Migration documentation:
  MIGRATION_PARTICLE_COLLOCATION.md
"""

import pytest


@pytest.mark.skip(
    reason="ParticleCollocationSolver has been moved to mfg-research repository. "
    "See MIGRATION_PARTICLE_COLLOCATION.md for details."
)
def test_particle_collocation_moved():
    """Placeholder test indicating migration to mfg-research."""


if __name__ == "__main__":
    print("=" * 70)
    print("SKIPPED: Particle-Collocation Integration Test")
    print("=" * 70)
    print()
    print("Particle-collocation methods have been moved to mfg-research repository.")
    print("This is part of architectural separation:")
    print("  - MFG_PDE: Stable infrastructure with classical methods")
    print("  - mfg-research: Novel research algorithms under active development")
    print()
    print("For particle-collocation usage, see:")
    print("  - mfg-research/algorithms/particle_collocation/")
    print("  - MIGRATION_PARTICLE_COLLOCATION.md")
    print()
