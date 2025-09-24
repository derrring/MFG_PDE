# MFG_PDE Scripts Directory

This directory contains utility scripts for development, environment management, verification, and system administration.

## Development Environment Setup

### `setup_development.sh` ⚡ **NEW**
Modern development environment setup supporting both UV and pip workflows.

**Usage:**
```bash
# Set up complete development environment (auto-detects UV or pip)
./scripts/setup_development.sh

# Verify existing installation
./scripts/setup_development.sh verify

# Show help
./scripts/setup_development.sh help
```

**Features:**
- ✅ Auto-detects UV for fast modern setup
- ✅ Falls back to traditional pip workflow
- ✅ Installs development dependencies
- ✅ Sets up pre-commit hooks
- ✅ Runs verification checks

## Development Tools

### `quick_type_check.py` ⚡ **NEW**
Rapid type checking and linting for development feedback.

**Usage:**
```bash
# Quick type check and linting
python scripts/quick_type_check.py

# From anywhere in project
./scripts/quick_type_check.py
```

**Features:**
- ✅ Fast mypy checking with error count
- ✅ Ruff linting integration
- ✅ Performance timing
- ✅ Focused output for rapid feedback

### `verify_modernization.py`
Comprehensive verification of package modernization and tool configuration.

**Usage:**
```bash
# Run full modernization verification
python scripts/verify_modernization.py
```

**Features:**
- ✅ pyproject.toml configuration verification
- ✅ Package import testing
- ✅ Typing modernization analysis
- ✅ Development tools validation

## Legacy Environment Management

### `manage_environments.sh`
Multi-environment management script for conda environments.

**Usage:**
```bash
# Create development environment
./scripts/manage_environments.sh create-dev

# Create performance-optimized environment
./scripts/manage_environments.sh create-performance

# Verify all environments
./scripts/manage_environments.sh verify

# Show environment information
./scripts/manage_environments.sh info
```

### `setup_env_vars.sh`
Environment variable configuration for development and research.

**Usage:**
```bash
# Source environment variables
source scripts/setup_env_vars.sh
```

## Quick Start Guide

For new developers:

```bash
# 1. Clone repository
git clone <repository-url>
cd MFG_PDE

# 2. Run modern setup (handles everything)
./scripts/setup_development.sh

# 3. Activate environment
source .venv/bin/activate

# 4. Quick verification
python scripts/quick_type_check.py

# 5. Start developing! 🚀
```

## Development Guidelines

- All scripts should be executable (`chmod +x`)
- Include usage documentation in script headers
- Use consistent error handling and logging
- Test scripts in clean environments before committing
- Prefer UV-based workflows for performance when available

## Script Maintenance

Scripts are maintained alongside the main codebase. Update documentation when adding new scripts or modifying existing functionality.

**Recent Updates:**
- ✅ Added modern UV-based development setup
- ✅ Added rapid type checking workflow
- ✅ Enhanced verification capabilities
