# MFG_PDE Scripts Directory

This directory contains utility scripts for development, environment management, and system administration.

## Environment Management

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

# Or include in conda environment activation
echo "source $(pwd)/scripts/setup_env_vars.sh" >> ~/.conda/envs/mfg_env_pde/etc/conda/activate.d/env_vars.sh
```

## Development Guidelines

- All scripts should be executable (`chmod +x`)
- Include usage documentation in script headers
- Use consistent error handling and logging
- Test scripts in clean environments before committing

## Script Maintenance

Scripts are maintained alongside the main codebase. Update documentation when adding new scripts or modifying existing functionality.