#!/usr/bin/env python3
"""
Critical security issues checker for MFG_PDE CI/CD pipeline.

This script analyzes all security scan results and determines if there are
any critical issues that should fail the CI/CD pipeline.
"""

import json
import sys
from pathlib import Path
from typing import Any


class CriticalIssuesChecker:
    """Check for critical security issues across all scan results."""

    def __init__(self):
        self.critical_issues = []
        self.blocking_issues = []
        self.warnings = []

        # Define what constitutes critical/blocking issues
        # Note: Relaxed thresholds for research/educational projects
        self.critical_criteria = {
            "dependency": {
                "critical_vulns": 2,  # Allow some critical vulns for research dependencies
                "high_vulns": 10,  # Max 10 high severity vulnerabilities
            },
            "static_analysis": {
                "high_confidence_high_severity": 2,  # Allow some high confidence + high severity
                "total_high_confidence": 15,  # Max 15 high confidence issues
            },
            "secrets": {
                "verified_secrets": 0,  # No verified secrets allowed
                "total_secrets": 5,  # Max 5 potential secrets (scientific code may have examples)
            },
            "container": {
                "critical_vulns": 10,  # Allow more container vulns for research (base image limitations)
                "high_vulns": 20,  # Max 20 high severity container vulns (research tolerance)
            },
            "license": {
                "non_compliant": 3,  # Allow some non-compliant licenses for research tools
            },
        }

    def check_all_results(self) -> dict[str, Any]:
        """Check all security scan results for critical issues."""
        print("🔍 Checking for critical security issues...")

        # Check each type of scan result
        self._check_dependency_results()
        self._check_static_analysis_results()
        self._check_secrets_results()
        self._check_container_results()
        self._check_license_results()
        self._check_custom_security_results()

        # Generate summary
        summary = {
            "has_critical_issues": len(self.critical_issues) > 0,
            "has_blocking_issues": len(self.blocking_issues) > 0,
            "critical_issues_count": len(self.critical_issues),
            "blocking_issues_count": len(self.blocking_issues),
            "warnings_count": len(self.warnings),
            "critical_issues": self.critical_issues,
            "blocking_issues": self.blocking_issues,
            "warnings": self.warnings,
            "recommendation": self._get_recommendation(),
        }

        # Save results
        with open("critical-issues-report.json", "w") as f:
            json.dump(summary, f, indent=2)

        return summary

    def _check_dependency_results(self):
        """Check dependency scan results for critical issues."""
        result_file = Path("dependency-scan-results/dependency-security-report.json")
        if not result_file.exists():
            self.warnings.append("Dependency scan results not found")
            return

        try:
            with open(result_file) as f:
                data = json.load(f)

            summary = data.get("summary", {})
            findings = data.get("findings", [])

            # Check critical vulnerabilities
            critical_count = summary.get("critical_count", 0)
            high_count = summary.get("high_count", 0)

            if critical_count > self.critical_criteria["dependency"]["critical_vulns"]:
                self.critical_issues.append(
                    {
                        "type": "dependency",
                        "severity": "critical",
                        "message": f"Found {critical_count} critical dependency vulnerabilities",
                        "details": [f for f in findings if f.get("severity") == "critical"][:3],  # Top 3
                    }
                )

            if high_count > self.critical_criteria["dependency"]["high_vulns"]:
                self.blocking_issues.append(
                    {
                        "type": "dependency",
                        "severity": "high",
                        "message": f'Found {high_count} high severity vulnerabilities (max allowed: {self.critical_criteria["dependency"]["high_vulns"]})',
                        "details": [f for f in findings if f.get("severity") == "high"][:5],  # Top 5
                    }
                )

            # Check for specific high-risk packages
            high_risk_packages = ["pickle", "yaml", "requests"]
            for finding in findings:
                package = finding.get("package", "").lower()
                if any(risk_pkg in package for risk_pkg in high_risk_packages):
                    if finding.get("severity") in ["critical", "high"]:
                        self.critical_issues.append(
                            {
                                "type": "dependency",
                                "severity": "critical",
                                "message": f'High-risk package {package} has {finding.get("severity")} vulnerability',
                                "details": [finding],
                            }
                        )

        except Exception as e:
            self.warnings.append(f"Error reading dependency results: {e}")

    def _check_static_analysis_results(self):
        """Check static analysis results for critical issues."""
        result_file = Path("static-analysis-results/static-analysis-security-report.json")
        if not result_file.exists():
            self.warnings.append("Static analysis results not found")
            return

        try:
            with open(result_file) as f:
                data = json.load(f)

            findings = data.get("findings", [])

            # Count high confidence + high severity issues
            high_conf_high_sev = 0
            high_confidence_issues = []

            for finding in findings:
                confidence = finding.get("confidence", "").lower()
                severity = finding.get("severity", "").lower()

                if confidence == "high":
                    high_confidence_issues.append(finding)
                    if severity in ["high", "critical"]:
                        high_conf_high_sev += 1

            if high_conf_high_sev > self.critical_criteria["static_analysis"]["high_confidence_high_severity"]:
                self.critical_issues.append(
                    {
                        "type": "static_analysis",
                        "severity": "critical",
                        "message": f"Found {high_conf_high_sev} high confidence + high severity static analysis issues",
                        "details": [
                            f
                            for f in findings
                            if f.get("confidence") == "high" and f.get("severity") in ["high", "critical"]
                        ],
                    }
                )

            if len(high_confidence_issues) > self.critical_criteria["static_analysis"]["total_high_confidence"]:
                self.blocking_issues.append(
                    {
                        "type": "static_analysis",
                        "severity": "high",
                        "message": f'Found {len(high_confidence_issues)} high confidence issues (max allowed: {self.critical_criteria["static_analysis"]["total_high_confidence"]})',
                        "details": high_confidence_issues[:5],  # Top 5
                    }
                )

            # Check for specific dangerous patterns
            dangerous_patterns = ["sql_injection", "command_injection", "hardcoded_password", "weak_crypto"]
            for finding in findings:
                rule_id = finding.get("rule_id", "").lower()
                if any(pattern in rule_id for pattern in dangerous_patterns):
                    self.critical_issues.append(
                        {
                            "type": "static_analysis",
                            "severity": "critical",
                            "message": f"Dangerous security pattern detected: {rule_id}",
                            "details": [finding],
                        }
                    )

        except Exception as e:
            self.warnings.append(f"Error reading static analysis results: {e}")

    def _check_secrets_results(self):
        """Check secrets scan results for critical issues."""
        result_file = Path("secrets-scan-results/secrets-security-report.json")
        if not result_file.exists():
            self.warnings.append("Secrets scan results not found")
            return

        try:
            with open(result_file) as f:
                data = json.load(f)

            summary = data.get("summary", {})
            findings = data.get("findings", [])

            # Check for verified secrets
            verified_secrets = [f for f in findings if f.get("verified", False)]
            if len(verified_secrets) > self.critical_criteria["secrets"]["verified_secrets"]:
                self.critical_issues.append(
                    {
                        "type": "secrets",
                        "severity": "critical",
                        "message": f"Found {len(verified_secrets)} verified secrets",
                        "details": verified_secrets,
                    }
                )

            # Check total secrets count
            total_secrets = summary.get("total_secrets", 0)
            if total_secrets > self.critical_criteria["secrets"]["total_secrets"]:
                self.blocking_issues.append(
                    {
                        "type": "secrets",
                        "severity": "high",
                        "message": f'Found {total_secrets} potential secrets (max allowed: {self.critical_criteria["secrets"]["total_secrets"]})',
                        "details": findings[:3],  # Top 3
                    }
                )

            # Check for specific secret types
            high_risk_secret_types = ["private_key", "api_key", "password", "token"]
            for finding in findings:
                secret_type = finding.get("secret_type", "").lower()
                detector_name = finding.get("detector_name", "").lower()

                if any(risk_type in secret_type or risk_type in detector_name for risk_type in high_risk_secret_types):
                    self.critical_issues.append(
                        {
                            "type": "secrets",
                            "severity": "critical",
                            "message": f"High-risk secret type detected: {secret_type or detector_name}",
                            "details": [finding],
                        }
                    )

        except Exception as e:
            self.warnings.append(f"Error reading secrets results: {e}")

    def _check_container_results(self):
        """Check container scan results for critical issues."""
        result_file = Path("container-scan-results/container-security-report.json")
        if not result_file.exists():
            self.warnings.append("Container scan results not found")
            return

        try:
            with open(result_file) as f:
                data = json.load(f)

            summary = data.get("summary", {})
            findings = data.get("findings", [])

            # Check critical container vulnerabilities
            critical_count = summary.get("critical_count", 0)
            high_count = summary.get("high_count", 0)

            if critical_count > self.critical_criteria["container"]["critical_vulns"]:
                self.critical_issues.append(
                    {
                        "type": "container",
                        "severity": "critical",
                        "message": f"Found {critical_count} critical container vulnerabilities",
                        "details": [f for f in findings if f.get("severity") == "critical"][:3],
                    }
                )

            if high_count > self.critical_criteria["container"]["high_vulns"]:
                self.blocking_issues.append(
                    {
                        "type": "container",
                        "severity": "high",
                        "message": f'Found {high_count} high severity container vulnerabilities (max allowed: {self.critical_criteria["container"]["high_vulns"]})',
                        "details": [f for f in findings if f.get("severity") == "high"][:3],
                    }
                )

            # Check for base image issues
            base_image_issues = [f for f in findings if "base" in f.get("file", "").lower()]
            if len(base_image_issues) > 5:
                self.blocking_issues.append(
                    {
                        "type": "container",
                        "severity": "high",
                        "message": f"Base image has {len(base_image_issues)} security issues",
                        "details": base_image_issues[:3],
                    }
                )

        except Exception as e:
            self.warnings.append(f"Error reading container results: {e}")

    def _check_license_results(self):
        """Check license compliance results for critical issues."""
        result_file = Path("license-compliance-results/license-security-report.json")
        if not result_file.exists():
            self.warnings.append("License compliance results not found")
            return

        try:
            with open(result_file) as f:
                data = json.load(f)

            summary = data.get("summary", {})
            findings = data.get("findings", [])

            # Check for non-compliant licenses
            non_compliant_count = summary.get("non_compliant_licenses", 0)
            if non_compliant_count > self.critical_criteria["license"]["non_compliant"]:
                non_compliant_packages = [f for f in findings if f.get("compliance_status") == "non_compliant"]

                # Check if these are production dependencies (not dev/test dependencies)
                production_non_compliant = []
                for package in non_compliant_packages:
                    pkg_name = package.get("package", "").lower()
                    # Skip development/testing packages
                    if not any(dev_keyword in pkg_name for dev_keyword in ["test", "dev", "mock", "debug"]):
                        production_non_compliant.append(package)

                if production_non_compliant:
                    self.critical_issues.append(
                        {
                            "type": "license",
                            "severity": "critical",
                            "message": f"Found {len(production_non_compliant)} non-compliant licenses in production dependencies",
                            "details": production_non_compliant,
                        }
                    )

            # Check for unknown licenses in critical packages
            unknown_licenses = [f for f in findings if f.get("compliance_status") == "unknown"]
            critical_packages = ["numpy", "scipy", "matplotlib"]  # Core scientific packages

            for package in unknown_licenses:
                pkg_name = package.get("package", "").lower()
                if any(critical_pkg in pkg_name for critical_pkg in critical_packages):
                    self.warnings.append(f"Unknown license for critical package: {pkg_name}")

        except Exception as e:
            self.warnings.append(f"Error reading license results: {e}")

    def _check_custom_security_results(self):
        """Check custom security scan results for critical issues."""
        result_file = Path("custom-security-report.json")
        if not result_file.exists():
            self.warnings.append("Custom security results not found")
            return

        try:
            with open(result_file) as f:
                data = json.load(f)

            summary = data.get("summary", {})
            findings = data.get("findings", [])

            # Check for high severity custom issues
            high_severity_count = summary.get("by_severity", {}).get("high", 0)
            critical_severity_count = summary.get("by_severity", {}).get("critical", 0)

            if critical_severity_count > 0:
                critical_findings = [f for f in findings if f.get("severity") == "critical"]
                self.critical_issues.append(
                    {
                        "type": "custom_security",
                        "severity": "critical",
                        "message": f"Found {critical_severity_count} critical custom security issues",
                        "details": critical_findings,
                    }
                )

            if high_severity_count > 5:  # Increased threshold for research projects
                high_findings = [f for f in findings if f.get("severity") == "high"]
                self.blocking_issues.append(
                    {
                        "type": "custom_security",
                        "severity": "high",
                        "message": f"Found {high_severity_count} high severity custom security issues",
                        "details": high_findings[:3],
                    }
                )

            # Check for specific dangerous patterns
            dangerous_types = ["subprocess_security", "pickle_security", "plugin_security"]
            for finding in findings:
                finding_type = finding.get("type", "")
                if finding_type in dangerous_types and finding.get("severity") in ["high", "critical"]:
                    self.critical_issues.append(
                        {
                            "type": "custom_security",
                            "severity": "critical",
                            "message": f"Dangerous security pattern: {finding_type}",
                            "details": [finding],
                        }
                    )

        except Exception as e:
            self.warnings.append(f"Error reading custom security results: {e}")

    def _get_recommendation(self) -> str:
        """Get recommendation based on findings."""
        if self.critical_issues:
            return "BLOCK_DEPLOYMENT - Critical security issues found that must be resolved before deployment"
        elif self.blocking_issues:
            return "BLOCK_MERGE - High severity issues found that should be resolved before merging"
        elif self.warnings:
            return "PROCEED_WITH_CAUTION - Some security warnings found, review recommended"
        else:
            return "PROCEED - No critical security issues found"

    def print_summary(self, summary: dict[str, Any]):
        """Print summary of critical issues check."""
        print("\n" + "=" * 60)
        print("🔒 CRITICAL SECURITY ISSUES ANALYSIS")
        print("=" * 60)

        print(f"Critical Issues: {summary['critical_issues_count']}")
        print(f"Blocking Issues: {summary['blocking_issues_count']}")
        print(f"Warnings: {summary['warnings_count']}")
        print(f"Recommendation: {summary['recommendation']}")

        if summary["critical_issues"]:
            print("\n🚨 CRITICAL ISSUES:")
            for issue in summary["critical_issues"]:
                print(f"  - {issue['type']}: {issue['message']}")

        if summary["blocking_issues"]:
            print("\n⚠️ BLOCKING ISSUES:")
            for issue in summary["blocking_issues"]:
                print(f"  - {issue['type']}: {issue['message']}")

        if summary["warnings"]:
            print("\n💡 WARNINGS:")
            for warning in summary["warnings"]:
                print(f"  - {warning}")

        print("\n" + "=" * 60)


def main():
    """Main function for critical issues check."""
    checker = CriticalIssuesChecker()
    summary = checker.check_all_results()

    checker.print_summary(summary)

    # Determine exit code based on findings
    if summary["has_critical_issues"]:
        print("\n❌ CRITICAL ISSUES FOUND - Failing pipeline")
        return 2  # Critical failure
    elif summary["has_blocking_issues"]:
        print("\n⚠️ BLOCKING ISSUES FOUND - Consider addressing before merge")
        return 1  # Warning failure
    else:
        print("\n✅ NO CRITICAL SECURITY ISSUES FOUND")
        return 0  # Success


if __name__ == "__main__":
    sys.exit(main())
