#!/usr/bin/env python3
"""
Custom security checks for MFG_PDE scientific computing package.

This script performs domain-specific security checks that are relevant
for scientific computing packages but may not be covered by general
security scanning tools.
"""

import ast
import json
import re
import sys
from pathlib import Path
from typing import Any


class CustomSecurityChecker:
    """Custom security checks for scientific computing packages."""

    def __init__(self):
        self.findings = []
        self.package_root = Path("mfg_pde")
        self.test_root = Path("tests")

    def run_all_checks(self) -> dict[str, Any]:
        """Run all custom security checks."""
        print("🔍 Running custom security checks for MFG_PDE...")

        # Scientific computing specific checks
        self.check_numerical_stability_patterns()
        self.check_memory_management_patterns()
        self.check_input_validation_patterns()
        self.check_subprocess_usage()
        self.check_pickle_security()
        self.check_file_operations()
        self.check_configuration_security()
        self.check_plugin_security()
        self.check_test_security_patterns()

        # Generate report
        report = {
            "timestamp": self._get_timestamp(),
            "total_findings": len(self.findings),
            "findings": self.findings,
            "summary": self._generate_summary(),
            "recommendations": self._generate_recommendations(),
        }

        # Save report
        with open("custom-security-report.json", "w") as f:
            json.dump(report, f, indent=2)

        print(f"✅ Custom security checks completed. Found {len(self.findings)} issues.")
        return report

    def check_numerical_stability_patterns(self):
        """Check for numerical stability and overflow patterns."""
        print("🔢 Checking numerical stability patterns...")

        dangerous_patterns = [
            (r"np\.power\([^,]+,\s*[^)]*\)", "Use np.power with caution - potential overflow"),
            (r"np\.exp\([^)]*\)", "Check for potential overflow in np.exp"),
            (r"1\s*/\s*[a-zA-Z_]", "Division by variable - check for zero division"),
            (r"np\.linalg\.inv\([^)]*\)", "Matrix inversion - check for singular matrices"),
            (r"np\.sqrt\([^)]*\)", "Check for negative inputs to sqrt"),
            (r"np\.log\([^)]*\)", "Check for zero/negative inputs to log"),
        ]

        for py_file in self.package_root.rglob("*.py"):
            try:
                content = py_file.read_text()
                for pattern, message in dangerous_patterns:
                    matches = re.finditer(pattern, content)
                    for match in matches:
                        line_num = content[: match.start()].count("\n") + 1
                        self.findings.append(
                            {
                                "type": "numerical_stability",
                                "severity": "medium",
                                "file": str(py_file),
                                "line": line_num,
                                "pattern": pattern,
                                "message": message,
                                "code_snippet": self._get_code_snippet(content, line_num),
                            }
                        )
            except Exception as e:
                print(f"Warning: Could not read {py_file}: {e}")

    def check_memory_management_patterns(self):
        """Check for potential memory management issues."""
        print("💾 Checking memory management patterns...")

        memory_patterns = [
            (r"np\.zeros\([^)]*\)\s*\*", "Inefficient memory allocation pattern"),
            (r"np\.concatenate\([^)]*\)", "Potential memory fragmentation in loops"),
            (r"\.append\([^)]*\)", "List append in loop - consider pre-allocation"),
            (r"range\(.*len\([^)]*\)\)", "Consider vectorized operations instead of loops"),
        ]

        for py_file in self.package_root.rglob("*.py"):
            try:
                content = py_file.read_text()

                # Check for large array allocations without bounds checking
                large_array_pattern = r"np\.(zeros|ones|empty|full)\(\s*\([^)]*\d{6,}[^)]*\)"
                matches = re.finditer(large_array_pattern, content)
                for match in matches:
                    line_num = content[: match.start()].count("\n") + 1
                    self.findings.append(
                        {
                            "type": "memory_management",
                            "severity": "medium",
                            "file": str(py_file),
                            "line": line_num,
                            "message": "Large array allocation without size validation",
                            "code_snippet": self._get_code_snippet(content, line_num),
                        }
                    )

                # Check other memory patterns
                for pattern, message in memory_patterns:
                    matches = re.finditer(pattern, content)
                    for match in matches:
                        line_num = content[: match.start()].count("\n") + 1
                        self.findings.append(
                            {
                                "type": "memory_management",
                                "severity": "low",
                                "file": str(py_file),
                                "line": line_num,
                                "pattern": pattern,
                                "message": message,
                                "code_snippet": self._get_code_snippet(content, line_num),
                            }
                        )

            except Exception as e:
                print(f"Warning: Could not read {py_file}: {e}")

    def check_input_validation_patterns(self):
        """Check for proper input validation in scientific functions."""
        print("🛡️ Checking input validation patterns...")

        for py_file in self.package_root.rglob("*.py"):
            try:
                content = py_file.read_text()
                tree = ast.parse(content)

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        self._check_function_input_validation(node, py_file, content)

            except Exception as e:
                print(f"Warning: Could not parse {py_file}: {e}")

    def check_subprocess_usage(self):
        """Check for potentially unsafe subprocess usage."""
        print("⚙️ Checking subprocess usage...")

        dangerous_subprocess_patterns = [
            (r"subprocess\.call\([^)]*shell\s*=\s*True", "subprocess.call with shell=True"),
            (r"subprocess\.run\([^)]*shell\s*=\s*True", "subprocess.run with shell=True"),
            (r"os\.system\([^)]*\)", "os.system usage"),
            (r"os\.popen\([^)]*\)", "os.popen usage"),
        ]

        for py_file in self.package_root.rglob("*.py"):
            try:
                content = py_file.read_text()
                for pattern, message in dangerous_subprocess_patterns:
                    matches = re.finditer(pattern, content)
                    for match in matches:
                        line_num = content[: match.start()].count("\n") + 1

                        # Check if this is in a test or example file - lower severity
                        is_test_or_example = any(
                            keyword in str(py_file).lower() for keyword in ["test", "example", "demo"]
                        )
                        severity = "medium" if is_test_or_example else "high"

                        self.findings.append(
                            {
                                "type": "subprocess_security",
                                "severity": severity,
                                "file": str(py_file),
                                "line": line_num,
                                "pattern": pattern,
                                "message": f"Potentially unsafe {message}",
                                "code_snippet": self._get_code_snippet(content, line_num),
                            }
                        )
            except Exception as e:
                print(f"Warning: Could not read {py_file}: {e}")

    def check_pickle_security(self):
        """Check for pickle security issues."""
        print("🥒 Checking pickle security...")

        pickle_patterns = [
            (r"pickle\.load\([^)]*\)", "Unsafe pickle.load usage"),
            (r"pickle\.loads\([^)]*\)", "Unsafe pickle.loads usage"),
            (r"dill\.load\([^)]*\)", "Unsafe dill.load usage"),
            (r"joblib\.load\([^)]*\)", "Check joblib.load for trusted sources only"),
        ]

        for py_file in self.package_root.rglob("*.py"):
            try:
                content = py_file.read_text()
                for pattern, message in pickle_patterns:
                    matches = re.finditer(pattern, content)
                    for match in matches:
                        line_num = content[: match.start()].count("\n") + 1

                        # Check if there's input validation nearby
                        surrounding_context = self._get_surrounding_context(content, line_num, 5)
                        has_validation = any(
                            keyword in surrounding_context.lower()
                            for keyword in ["validate", "check", "verify", "trust"]
                        )

                        severity = "medium" if has_validation else "high"
                        self.findings.append(
                            {
                                "type": "pickle_security",
                                "severity": severity,
                                "file": str(py_file),
                                "line": line_num,
                                "pattern": pattern,
                                "message": f"{message} - ensure source is trusted",
                                "code_snippet": self._get_code_snippet(content, line_num),
                            }
                        )
            except Exception as e:
                print(f"Warning: Could not read {py_file}: {e}")

    def check_file_operations(self):
        """Check for potentially unsafe file operations."""
        print("📁 Checking file operations...")

        file_patterns = [
            (r"open\([^)]*[\'\"]\.\./.*[\'\"]\)", "Path traversal risk in file open"),
            (r"os\.path\.join\([^)]*\.\.[^)]*\)", "Potential path traversal"),
            (r"pathlib\.Path\([^)]*\.\.[^)]*\)", "Potential path traversal with pathlib"),
        ]

        for py_file in self.package_root.rglob("*.py"):
            try:
                content = py_file.read_text()
                for pattern, message in file_patterns:
                    matches = re.finditer(pattern, content)
                    for match in matches:
                        line_num = content[: match.start()].count("\n") + 1
                        self.findings.append(
                            {
                                "type": "file_security",
                                "severity": "medium",
                                "file": str(py_file),
                                "line": line_num,
                                "pattern": pattern,
                                "message": message,
                                "code_snippet": self._get_code_snippet(content, line_num),
                            }
                        )
            except Exception as e:
                print(f"Warning: Could not read {py_file}: {e}")

    def check_configuration_security(self):
        """Check for configuration security issues."""
        print("⚙️ Checking configuration security...")

        config_files = list(self.package_root.rglob("*config*.py"))

        for config_file in config_files:
            try:
                content = config_file.read_text()

                # Check for hardcoded credentials or sensitive data
                sensitive_patterns = [
                    (r"password\s*=\s*[\'\"]\w+[\'\"]+", "Hardcoded password"),
                    (r"api_key\s*=\s*[\'\"]\w+[\'\"]+", "Hardcoded API key"),
                    (r"secret\s*=\s*[\'\"]\w+[\'\"]+", "Hardcoded secret"),
                    (r"token\s*=\s*[\'\"]\w+[\'\"]+", "Hardcoded token"),
                ]

                for pattern, message in sensitive_patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        line_num = content[: match.start()].count("\n") + 1
                        self.findings.append(
                            {
                                "type": "config_security",
                                "severity": "high",
                                "file": str(config_file),
                                "line": line_num,
                                "pattern": pattern,
                                "message": f"{message} in configuration file",
                                "code_snippet": self._get_code_snippet(content, line_num),
                            }
                        )

            except Exception as e:
                print(f"Warning: Could not read {config_file}: {e}")

    def check_plugin_security(self):
        """Check for plugin system security issues."""
        print("🔌 Checking plugin security...")

        plugin_files = list(self.package_root.rglob("*plugin*.py"))

        for plugin_file in plugin_files:
            try:
                content = plugin_file.read_text()

                # Check for dynamic imports without validation
                dynamic_import_patterns = [
                    (r"importlib\.import_module\([^)]*\)", "Dynamic import without validation"),
                    (r"__import__\([^)]*\)", "Dynamic __import__ usage"),
                    (r"exec\([^)]*\)", "exec() usage in plugin system"),
                    (r"eval\([^)]*\)", "eval() usage in plugin system"),
                ]

                for pattern, message in dynamic_import_patterns:
                    matches = re.finditer(pattern, content)
                    for match in matches:
                        line_num = content[: match.start()].count("\n") + 1

                        # Check for validation nearby
                        surrounding_context = self._get_surrounding_context(content, line_num, 10)
                        has_validation = any(
                            keyword in surrounding_context.lower()
                            for keyword in ["validate", "check", "verify", "whitelist", "allowlist"]
                        )

                        severity = "medium" if has_validation else "high"
                        self.findings.append(
                            {
                                "type": "plugin_security",
                                "severity": severity,
                                "file": str(plugin_file),
                                "line": line_num,
                                "pattern": pattern,
                                "message": f"{message} - ensure proper validation",
                                "code_snippet": self._get_code_snippet(content, line_num),
                            }
                        )

            except Exception as e:
                print(f"Warning: Could not read {plugin_file}: {e}")

    def check_test_security_patterns(self):
        """Check for security anti-patterns in test files."""
        print("🧪 Checking test security patterns...")

        if not self.test_root.exists():
            return

        for test_file in self.test_root.rglob("*.py"):
            try:
                content = test_file.read_text()

                # Check for hardcoded sensitive data in tests
                test_security_patterns = [
                    (r"password\s*=\s*[\'\"]\w{8,}[\'\"]+", "Hardcoded password in test"),
                    (r"secret\s*=\s*[\'\"]\w{8,}[\'\"]+", "Hardcoded secret in test"),
                    (r"localhost:\d+", "Hardcoded localhost connection"),
                    (r"127\.0\.0\.1:\d+", "Hardcoded IP address"),
                ]

                for pattern, message in test_security_patterns:
                    matches = re.finditer(pattern, content)
                    for match in matches:
                        line_num = content[: match.start()].count("\n") + 1
                        self.findings.append(
                            {
                                "type": "test_security",
                                "severity": "low",
                                "file": str(test_file),
                                "line": line_num,
                                "pattern": pattern,
                                "message": f"{message} - use fixtures or environment variables",
                                "code_snippet": self._get_code_snippet(content, line_num),
                            }
                        )

            except Exception as e:
                print(f"Warning: Could not read {test_file}: {e}")

    def _check_function_input_validation(self, node: ast.FunctionDef, file_path: Path, content: str):
        """Check if function has proper input validation."""
        # Skip private functions and test functions
        if node.name.startswith("_") or "test" in str(file_path).lower():
            return

        # Look for functions that take numerical parameters
        has_numerical_params = False
        for arg in node.args.args:
            if any(hint in arg.arg.lower() for hint in ["array", "matrix", "data", "x", "y", "nx", "nt"]):
                has_numerical_params = True
                break

        if not has_numerical_params:
            return

        # Check if function body contains validation
        function_code = ast.get_source_segment(content, node)
        if function_code:
            validation_keywords = ["assert", "raise", "ValueError", "TypeError", "check", "validate"]
            has_validation = any(keyword in function_code for keyword in validation_keywords)

            if not has_validation:
                self.findings.append(
                    {
                        "type": "input_validation",
                        "severity": "medium",
                        "file": str(file_path),
                        "line": node.lineno,
                        "function": node.name,
                        "message": f"Function {node.name} may lack input validation for numerical parameters",
                        "code_snippet": function_code[:200] + "..." if len(function_code) > 200 else function_code,
                    }
                )

    def _get_code_snippet(self, content: str, line_num: int, context: int = 2) -> str:
        """Get code snippet around a specific line."""
        lines = content.split("\n")
        start = max(0, line_num - context - 1)
        end = min(len(lines), line_num + context)

        snippet_lines = []
        for i in range(start, end):
            marker = ">>>" if i == line_num - 1 else "   "
            snippet_lines.append(f"{marker} {i + 1:3d}: {lines[i]}")

        return "\n".join(snippet_lines)

    def _get_surrounding_context(self, content: str, line_num: int, context: int = 5) -> str:
        """Get surrounding context around a line."""
        lines = content.split("\n")
        start = max(0, line_num - context - 1)
        end = min(len(lines), line_num + context)
        return "\n".join(lines[start:end])

    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime

        return datetime.utcnow().isoformat()

    def _generate_summary(self) -> dict[str, Any]:
        """Generate summary of findings."""
        summary = {
            "total_findings": len(self.findings),
            "by_severity": {"critical": 0, "high": 0, "medium": 0, "low": 0},
            "by_type": {},
            "files_with_issues": len({f["file"] for f in self.findings}),
        }

        for finding in self.findings:
            # Count by severity
            severity = finding.get("severity", "unknown")
            if severity in summary["by_severity"]:
                summary["by_severity"][severity] += 1

            # Count by type
            finding_type = finding.get("type", "unknown")
            summary["by_type"][finding_type] = summary["by_type"].get(finding_type, 0) + 1

        return summary

    def _generate_recommendations(self) -> list[str]:
        """Generate security recommendations based on findings."""
        recommendations = []

        if not self.findings:
            recommendations.append("✅ No custom security issues found")
            return recommendations

        # Type-specific recommendations
        types_found = {f.get("type", "unknown") for f in self.findings}

        if "numerical_stability" in types_found:
            recommendations.append("🔢 Add input validation for numerical functions (range checks, NaN/Inf handling)")
            recommendations.append("📊 Implement numerical stability tests with edge cases")

        if "memory_management" in types_found:
            recommendations.append("💾 Review memory allocation patterns for large arrays")
            recommendations.append("📏 Add memory usage monitoring for computationally intensive functions")

        if "subprocess_security" in types_found:
            recommendations.append("⚙️ Avoid shell=True in subprocess calls")
            recommendations.append("🔒 Use subprocess with explicit argument lists")

        if "pickle_security" in types_found:
            recommendations.append("🥒 Validate pickle sources and consider safer serialization formats")
            recommendations.append("🔐 Implement signature verification for serialized data")

        if "plugin_security" in types_found:
            recommendations.append("🔌 Add plugin validation and sandboxing")
            recommendations.append("📋 Implement plugin allowlist/blocklist mechanisms")

        if "input_validation" in types_found:
            recommendations.append("🛡️ Add comprehensive input validation to public functions")
            recommendations.append("📝 Document expected input ranges and formats")

        # General recommendations
        recommendations.append("🔍 Consider adding runtime parameter validation")
        recommendations.append("📋 Implement security guidelines for contributors")
        recommendations.append("🧪 Add security-focused test cases")

        return recommendations


def main():
    """Main function for custom security checks."""
    checker = CustomSecurityChecker()
    report = checker.run_all_checks()

    # Print summary
    summary = report["summary"]
    print("\n📊 Custom Security Check Summary:")
    print(f"  Total findings: {summary['total_findings']}")
    print(f"  Files with issues: {summary['files_with_issues']}")
    print(f"  By severity: {summary['by_severity']}")
    print(f"  By type: {summary['by_type']}")

    # Return appropriate exit code (adjusted for research projects)
    critical_severity_count = summary["by_severity"].get("critical", 0)
    high_severity_count = summary["by_severity"].get("high", 0)

    if critical_severity_count > 0:
        print(f"\n❌ Found {critical_severity_count} critical severity issues")
        return 1  # Non-zero exit code for critical issues
    elif high_severity_count > 3:  # Allow up to 3 high severity issues for research projects
        print(f"\n⚠️ Found {high_severity_count} high severity issues (threshold: 3)")
        return 1  # Non-zero exit code for too many high severity issues
    else:
        print(
            f"\n✅ Security check passed ({high_severity_count} high, {critical_severity_count} critical - within research project tolerances)"
        )
        return 0


if __name__ == "__main__":
    sys.exit(main())
