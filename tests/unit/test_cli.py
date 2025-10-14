#!/usr/bin/env python3
"""
Unit tests for mfg_pde/utils/cli.py

Tests CLI utilities including:
- Argument parser creation and configuration
- Config file loading (JSON/YAML)
- Config file saving
- Config merging and conversion
- Args to config conversion
- CLI subcommands
- Error handling
"""

import argparse
import json
import tempfile
from pathlib import Path

import pytest

from mfg_pde.utils.cli import (
    YAML_AVAILABLE,
    args_to_config,
    create_base_parser,
    create_solver_cli,
    load_config_file,
    merge_configs,
    save_config_file,
)

# ===================================================================
# Test Parser Creation
# ===================================================================


@pytest.mark.unit
def test_create_base_parser_returns_parser():
    """Test create_base_parser returns ArgumentParser instance."""
    parser = create_base_parser()
    assert isinstance(parser, argparse.ArgumentParser)


@pytest.mark.unit
def test_create_base_parser_has_description():
    """Test parser has description."""
    parser = create_base_parser()
    assert "MFG" in parser.description
    assert "Mean Field Games" in parser.description


@pytest.mark.unit
def test_create_base_parser_has_epilog():
    """Test parser has epilog with repository link."""
    parser = create_base_parser()
    assert parser.epilog is not None
    assert "github" in parser.epilog.lower()


@pytest.mark.unit
def test_create_base_parser_default_values():
    """Test parser has correct default values."""
    parser = create_base_parser()
    args = parser.parse_args([])

    # Problem defaults
    assert args.T == 1.0
    assert args.Nt == 50
    assert args.xmin == 0.0
    assert args.xmax == 1.0
    assert args.Nx == 100

    # Solver defaults
    assert args.solver_type == "fixed_point"
    assert args.preset == "balanced"

    # Execution defaults
    assert args.progress is True
    assert args.timing is True
    assert args.return_structured is True


@pytest.mark.unit
def test_create_base_parser_problem_arguments():
    """Test parser accepts problem configuration arguments."""
    parser = create_base_parser()
    args = parser.parse_args(["--T", "2.0", "--Nt", "100", "--Nx", "200"])

    assert args.T == 2.0
    assert args.Nt == 100
    assert args.Nx == 200


@pytest.mark.unit
def test_create_base_parser_solver_arguments():
    """Test parser accepts solver configuration arguments."""
    parser = create_base_parser()
    args = parser.parse_args(
        [
            "--solver-type",
            "fixed_point",
            "--preset",
            "accurate",
            "--max-iterations",
            "1000",
            "--tolerance",
            "1e-8",
        ]
    )

    assert args.solver_type == "fixed_point"
    assert args.preset == "accurate"
    assert args.max_iterations == 1000
    assert args.tolerance == 1e-8


@pytest.mark.unit
def test_create_base_parser_preset_choices():
    """Test parser accepts valid preset choices."""
    parser = create_base_parser()

    for preset in ["fast", "balanced", "accurate", "research"]:
        args = parser.parse_args(["--preset", preset])
        assert args.preset == preset


@pytest.mark.unit
def test_create_base_parser_invalid_preset_raises():
    """Test parser rejects invalid preset choices."""
    parser = create_base_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--preset", "invalid"])


@pytest.mark.unit
def test_create_base_parser_boolean_flags():
    """Test parser handles boolean flags correctly."""
    parser = create_base_parser()

    # Test verbose
    args = parser.parse_args(["--verbose"])
    assert args.verbose is True

    # Test quiet
    args = parser.parse_args(["--quiet"])
    assert args.quiet is True

    # Test no-progress
    args = parser.parse_args(["--no-progress"])
    assert args.progress is False

    # Test no-timing
    args = parser.parse_args(["--no-timing"])
    assert args.timing is False


@pytest.mark.unit
def test_create_base_parser_io_arguments():
    """Test parser accepts I/O arguments."""
    parser = create_base_parser()
    args = parser.parse_args(
        [
            "--config",
            "config.json",
            "--output",
            "results.json",
            "--save-config",
            "saved_config.json",
        ]
    )

    assert args.config == "config.json"
    assert args.output == "results.json"
    assert args.save_config == "saved_config.json"


@pytest.mark.unit
def test_create_base_parser_advanced_arguments():
    """Test parser accepts advanced arguments."""
    parser = create_base_parser()
    args = parser.parse_args(["--warm-start", "--profile"])

    assert args.warm_start is True
    assert args.profile is True


# ===================================================================
# Test Config File Loading
# ===================================================================


@pytest.mark.unit
def test_load_config_file_json():
    """Test loading JSON configuration file."""
    config_data = {
        "problem": {"T": 2.0, "Nt": 100},
        "solver": {"type": "fixed_point", "preset": "accurate"},
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config_data, f)
        temp_path = f.name

    try:
        loaded_config = load_config_file(temp_path)
        assert loaded_config == config_data
        assert loaded_config["problem"]["T"] == 2.0
        assert loaded_config["solver"]["type"] == "fixed_point"
    finally:
        Path(temp_path).unlink()


@pytest.mark.unit
@pytest.mark.skipif(not YAML_AVAILABLE, reason="PyYAML not available")
def test_load_config_file_yaml():
    """Test loading YAML configuration file."""
    config_data = {
        "problem": {"T": 2.0, "Nt": 100},
        "solver": {"type": "fixed_point", "preset": "accurate"},
    }

    import yaml

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(config_data, f)
        temp_path = f.name

    try:
        loaded_config = load_config_file(temp_path)
        assert loaded_config == config_data
    finally:
        Path(temp_path).unlink()


@pytest.mark.unit
def test_load_config_file_not_found():
    """Test loading nonexistent config file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError) as exc_info:
        load_config_file("nonexistent_config.json")

    assert "not found" in str(exc_info.value).lower()


@pytest.mark.unit
def test_load_config_file_unsupported_format():
    """Test loading unsupported format raises ValueError."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("invalid config")
        temp_path = f.name

    try:
        with pytest.raises(ValueError) as exc_info:
            load_config_file(temp_path)
        assert "unsupported" in str(exc_info.value).lower()
    finally:
        Path(temp_path).unlink()


@pytest.mark.unit
def test_load_config_file_invalid_json():
    """Test loading invalid JSON raises ValueError."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write("{invalid json")
        temp_path = f.name

    try:
        with pytest.raises(ValueError) as exc_info:
            load_config_file(temp_path)
        assert "error loading" in str(exc_info.value).lower()
    finally:
        Path(temp_path).unlink()


@pytest.mark.unit
def test_load_config_file_path_conversion():
    """Test load_config_file accepts both str and Path."""
    config_data = {"test": "value"}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config_data, f)
        temp_path = f.name

    try:
        # Test with string path
        config1 = load_config_file(temp_path)
        # Test with Path object
        config2 = load_config_file(Path(temp_path))

        assert config1 == config2 == config_data
    finally:
        Path(temp_path).unlink()


# ===================================================================
# Test Config File Saving
# ===================================================================


@pytest.mark.unit
def test_save_config_file_json(capsys):
    """Test saving configuration to JSON file."""
    config_data = {
        "problem": {"T": 2.0, "Nt": 100},
        "solver": {"type": "fixed_point"},
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        temp_path = f.name

    try:
        save_config_file(config_data, temp_path)

        # Check file was created
        assert Path(temp_path).exists()

        # Load and verify content
        with open(temp_path) as f:
            loaded = json.load(f)
        assert loaded == config_data

        # Check output message
        captured = capsys.readouterr()
        assert "saved" in captured.out.lower()
        assert temp_path in captured.out
    finally:
        Path(temp_path).unlink(missing_ok=True)


@pytest.mark.unit
@pytest.mark.skipif(not YAML_AVAILABLE, reason="PyYAML not available")
def test_save_config_file_yaml(capsys):
    """Test saving configuration to YAML file."""
    import yaml

    config_data = {
        "problem": {"T": 2.0, "Nt": 100},
        "solver": {"type": "fixed_point"},
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        temp_path = f.name

    try:
        save_config_file(config_data, temp_path)

        # Check file was created
        assert Path(temp_path).exists()

        # Load and verify content
        with open(temp_path) as f:
            loaded = yaml.safe_load(f)
        assert loaded == config_data

        # Check output message
        captured = capsys.readouterr()
        assert "saved" in captured.out.lower()
    finally:
        Path(temp_path).unlink(missing_ok=True)


@pytest.mark.unit
def test_save_config_file_default_to_json(capsys):
    """Test save defaults to JSON for unknown extensions."""
    config_data = {"test": "value"}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        temp_path = f.name

    try:
        save_config_file(config_data, temp_path)

        # Should save as JSON despite .txt extension
        with open(temp_path) as f:
            loaded = json.load(f)
        assert loaded == config_data
    finally:
        Path(temp_path).unlink(missing_ok=True)


@pytest.mark.unit
def test_save_config_file_error_handling(capsys):
    """Test save_config_file handles errors gracefully."""
    config_data = {"test": "value"}

    # Try to save to invalid path (directory)
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        invalid_path = Path(tmpdir)  # Directory, not file

        save_config_file(config_data, invalid_path)

        # Should print error message
        captured = capsys.readouterr()
        assert "error" in captured.out.lower()


# ===================================================================
# Test Config Merging
# ===================================================================


@pytest.mark.unit
def test_merge_configs_simple():
    """Test merging simple configurations."""
    base = {"a": 1, "b": 2}
    override = {"b": 3, "c": 4}

    merged = merge_configs(base, override)

    assert merged["a"] == 1  # From base
    assert merged["b"] == 3  # From override
    assert merged["c"] == 4  # From override


@pytest.mark.unit
def test_merge_configs_nested_dicts():
    """Test merging nested dictionaries."""
    base = {"solver": {"type": "fixed_point", "max_iterations": 100, "tolerance": 1e-6}}

    override = {"solver": {"max_iterations": 200}}

    merged = merge_configs(base, override)

    assert merged["solver"]["type"] == "fixed_point"  # Preserved from base
    assert merged["solver"]["max_iterations"] == 200  # Overridden
    assert merged["solver"]["tolerance"] == 1e-6  # Preserved from base


@pytest.mark.unit
def test_merge_configs_deep_nesting():
    """Test merging deeply nested configurations."""
    base = {"level1": {"level2": {"a": 1, "b": 2}, "c": 3}}

    override = {"level1": {"level2": {"b": 20}, "d": 4}}

    merged = merge_configs(base, override)

    assert merged["level1"]["level2"]["a"] == 1
    assert merged["level1"]["level2"]["b"] == 20
    assert merged["level1"]["c"] == 3
    assert merged["level1"]["d"] == 4


@pytest.mark.unit
def test_merge_configs_replaces_non_dict_values():
    """Test merge replaces non-dict values completely."""
    base = {"solver": {"type": "fixed_point", "params": [1, 2, 3]}}

    override = {"solver": {"params": [4, 5, 6]}}

    merged = merge_configs(base, override)

    # Lists are replaced, not merged
    assert merged["solver"]["params"] == [4, 5, 6]


@pytest.mark.unit
def test_merge_configs_preserves_base():
    """Test merge does not modify base config."""
    base = {"a": 1, "b": {"c": 2}}
    override = {"b": {"c": 3}}

    merged = merge_configs(base, override)

    # Base should be unchanged
    assert base["b"]["c"] == 2
    # Merged should have override value
    assert merged["b"]["c"] == 3


@pytest.mark.unit
def test_merge_configs_empty_override():
    """Test merging with empty override returns base."""
    base = {"a": 1, "b": 2}
    override = {}

    merged = merge_configs(base, override)

    assert merged == base


@pytest.mark.unit
def test_merge_configs_empty_base():
    """Test merging with empty base returns override."""
    base = {}
    override = {"a": 1, "b": 2}

    merged = merge_configs(base, override)

    assert merged == override


# ===================================================================
# Test Args to Config Conversion
# ===================================================================


@pytest.mark.unit
def test_args_to_config_default_values():
    """Test args_to_config with default argument values."""
    parser = create_base_parser()
    args = parser.parse_args([])

    config = args_to_config(args)

    # Check problem config
    assert config["problem"]["T"] == 1.0
    assert config["problem"]["Nt"] == 50
    assert config["problem"]["Nx"] == 100

    # Check solver config
    assert config["solver"]["type"] == "fixed_point"
    assert config["solver"]["preset"] == "balanced"

    # Check execution config
    assert config["execution"]["verbose"] is False
    assert config["execution"]["progress"] is True


@pytest.mark.unit
def test_args_to_config_custom_values():
    """Test args_to_config with custom argument values."""
    parser = create_base_parser()
    args = parser.parse_args(
        [
            "--T",
            "2.0",
            "--Nt",
            "100",
            "--solver-type",
            "fixed_point",
            "--preset",
            "accurate",
            "--max-iterations",
            "1000",
            "--tolerance",
            "1e-8",
        ]
    )

    config = args_to_config(args)

    assert config["problem"]["T"] == 2.0
    assert config["problem"]["Nt"] == 100
    assert config["solver"]["type"] == "fixed_point"
    assert config["solver"]["preset"] == "accurate"
    assert config["solver"]["max_iterations"] == 1000
    assert config["solver"]["tolerance"] == 1e-8


@pytest.mark.unit
def test_args_to_config_optional_parameters():
    """Test args_to_config omits None optional parameters."""
    parser = create_base_parser()
    args = parser.parse_args([])

    config = args_to_config(args)

    # Optional parameters should not be in config if None
    assert "max_iterations" not in config["solver"]
    assert "tolerance" not in config["solver"]
    assert "num_particles" not in config["solver"]


@pytest.mark.unit
def test_args_to_config_includes_optional_when_set():
    """Test args_to_config includes optional parameters when set."""
    parser = create_base_parser()
    args = parser.parse_args(["--max-iterations", "500", "--tolerance", "1e-7"])

    config = args_to_config(args)

    assert config["solver"]["max_iterations"] == 500
    assert config["solver"]["tolerance"] == 1e-7


@pytest.mark.unit
def test_args_to_config_verbose_quiet_interaction():
    """Test args_to_config handles verbose and quiet correctly."""
    parser = create_base_parser()

    # Verbose only
    args = parser.parse_args(["--verbose"])
    config = args_to_config(args)
    assert config["execution"]["verbose"] is True

    # Quiet overrides verbose
    args = parser.parse_args(["--verbose", "--quiet"])
    config = args_to_config(args)
    assert config["execution"]["verbose"] is False
    assert config["execution"]["progress"] is False
    assert config["execution"]["timing"] is False


@pytest.mark.unit
def test_args_to_config_progress_timing_flags():
    """Test args_to_config respects progress and timing flags."""
    parser = create_base_parser()

    # Default: progress and timing enabled
    args = parser.parse_args([])
    config = args_to_config(args)
    assert config["execution"]["progress"] is True
    assert config["execution"]["timing"] is True

    # Disable progress
    args = parser.parse_args(["--no-progress"])
    config = args_to_config(args)
    assert config["execution"]["progress"] is False

    # Disable timing
    args = parser.parse_args(["--no-timing"])
    config = args_to_config(args)
    assert config["execution"]["timing"] is False


@pytest.mark.unit
def test_args_to_config_io_parameters():
    """Test args_to_config includes I/O parameters."""
    parser = create_base_parser()
    args = parser.parse_args(["--output", "results.json", "--save-config", "config.json"])

    config = args_to_config(args)

    assert config["io"]["output"] == "results.json"
    assert config["io"]["save_config"] == "config.json"


@pytest.mark.unit
def test_args_to_config_structure():
    """Test args_to_config produces correct structure."""
    parser = create_base_parser()
    args = parser.parse_args([])

    config = args_to_config(args)

    # Check top-level keys
    assert "problem" in config
    assert "solver" in config
    assert "execution" in config
    assert "io" in config

    # Check nested structure
    assert isinstance(config["problem"], dict)
    assert isinstance(config["solver"], dict)
    assert isinstance(config["execution"], dict)
    assert isinstance(config["io"], dict)


# ===================================================================
# Test Solver CLI Creation
# ===================================================================


@pytest.mark.unit
def test_create_solver_cli_returns_parser():
    """Test create_solver_cli returns ArgumentParser."""
    parser = create_solver_cli()
    assert isinstance(parser, argparse.ArgumentParser)


@pytest.mark.unit
def test_create_solver_cli_has_subcommands():
    """Test create_solver_cli creates subcommands."""
    parser = create_solver_cli()

    # Parse with solve subcommand
    args = parser.parse_args(["solve"])
    assert args.command == "solve"

    # Parse with config subcommand
    args = parser.parse_args(["config", "generate", "output.json"])
    assert args.command == "config"


@pytest.mark.unit
def test_create_solver_cli_solve_command():
    """Test solve subcommand accepts problem file."""
    parser = create_solver_cli()

    args = parser.parse_args(["solve", "problem.py", "--problem-class", "MyProblem"])

    assert args.command == "solve"
    assert args.problem_file == "problem.py"
    assert args.problem_class == "MyProblem"


@pytest.mark.unit
def test_create_solver_cli_solve_command_defaults():
    """Test solve subcommand has correct defaults."""
    parser = create_solver_cli()

    args = parser.parse_args(["solve"])

    assert args.command == "solve"
    assert args.problem_file is None
    assert args.problem_class == "MFGProblem"


@pytest.mark.unit
def test_create_solver_cli_config_generate():
    """Test config generate subcommand."""
    parser = create_solver_cli()

    args = parser.parse_args(["config", "generate", "output.json", "--format", "json"])

    assert args.command == "config"
    assert args.config_command == "generate"
    assert args.output_file == "output.json"
    assert args.format == "json"


@pytest.mark.unit
def test_create_solver_cli_config_validate():
    """Test config validate subcommand."""
    parser = create_solver_cli()

    args = parser.parse_args(["config", "validate", "config.json"])

    assert args.command == "config"
    assert args.config_command == "validate"
    assert args.config_file == "config.json"


# ===================================================================
# Test Integration Scenarios
# ===================================================================


@pytest.mark.unit
def test_full_workflow_args_to_config_to_file():
    """Test full workflow: args → config → save → load."""
    parser = create_base_parser()
    args = parser.parse_args(["--T", "3.0", "--Nt", "150", "--preset", "research"])

    # Convert args to config
    config = args_to_config(args)

    # Save config
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        temp_path = f.name

    try:
        save_config_file(config, temp_path)

        # Load config back
        loaded_config = load_config_file(temp_path)

        # Verify round-trip
        assert loaded_config["problem"]["T"] == 3.0
        assert loaded_config["problem"]["Nt"] == 150
        assert loaded_config["solver"]["preset"] == "research"
    finally:
        Path(temp_path).unlink(missing_ok=True)


@pytest.mark.unit
def test_config_override_workflow():
    """Test workflow: load config → override with args → merge."""
    # Create base config file
    base_config = {"problem": {"T": 1.0, "Nt": 50}, "solver": {"preset": "balanced"}}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(base_config, f)
        temp_path = f.name

    try:
        # Load base config
        loaded = load_config_file(temp_path)

        # Create override config from args
        parser = create_base_parser()
        args = parser.parse_args(["--T", "2.0", "--preset", "accurate"])
        override_config = args_to_config(args)

        # Merge
        final_config = merge_configs(loaded, override_config)

        # Verify merge
        assert final_config["problem"]["T"] == 2.0  # Overridden
        assert final_config["problem"]["Nt"] == 50  # From base
        assert final_config["solver"]["preset"] == "accurate"  # Overridden
    finally:
        Path(temp_path).unlink()


# ===================================================================
# Test YAML Availability Handling
# ===================================================================


@pytest.mark.unit
def test_yaml_available_flag():
    """Test YAML_AVAILABLE flag is boolean."""
    assert isinstance(YAML_AVAILABLE, bool)


@pytest.mark.unit
@pytest.mark.skipif(YAML_AVAILABLE, reason="Test requires YAML unavailable")
def test_load_yaml_without_pyyaml_raises():
    """Test loading YAML without PyYAML raises helpful error."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("test: value")
        temp_path = f.name

    try:
        with pytest.raises(ValueError) as exc_info:
            load_config_file(temp_path)
        assert "PyYAML" in str(exc_info.value)
    finally:
        Path(temp_path).unlink()


# ===================================================================
# Test Module Exports
# ===================================================================


@pytest.mark.unit
def test_module_exports_all_functions():
    """Test module exports all expected functions."""
    from mfg_pde.utils import cli

    assert hasattr(cli, "create_base_parser")
    assert hasattr(cli, "load_config_file")
    assert hasattr(cli, "save_config_file")
    assert hasattr(cli, "merge_configs")
    assert hasattr(cli, "args_to_config")
    assert hasattr(cli, "create_solver_cli")


@pytest.mark.unit
def test_module_has_docstring():
    """Test module has comprehensive docstring."""
    from mfg_pde.utils import cli

    assert cli.__doc__ is not None
    assert "Command Line Interface" in cli.__doc__
    assert "MFG" in cli.__doc__
