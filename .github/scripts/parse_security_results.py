#!/usr/bin/env python3
"""
Security results parser for MFG_PDE CI/CD pipeline.

This script parses various security scan results and generates
unified reports for the security dashboard.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


class SecurityResultsParser:
    """Parse and consolidate security scan results."""

    def __init__(self):
        self.timestamp = datetime.utcnow().isoformat()
        self.results = {
            "timestamp": self.timestamp,
            "scan_type": None,
            "summary": {},
            "findings": [],
            "recommendations": [],
        }

    def parse_dependency_results(self) -> dict[str, Any]:
        """Parse dependency scan results from Safety and pip-audit."""
        print("📋 Parsing dependency scan results...")

        findings = []
        summary = {
            "total_vulnerabilities": 0,
            "critical_count": 0,
            "high_count": 0,
            "medium_count": 0,
            "low_count": 0,
            "tools_used": ["safety", "pip-audit"],
        }

        # Parse Safety results
        safety_findings = self._parse_safety_results()
        findings.extend(safety_findings)

        # Parse pip-audit results
        pip_audit_findings = self._parse_pip_audit_results()
        findings.extend(pip_audit_findings)

        # Update summary counts
        for finding in findings:
            severity = finding.get("severity", "unknown").lower()
            summary["total_vulnerabilities"] += 1

            if severity in ["critical"]:
                summary["critical_count"] += 1
            elif severity in ["high"]:
                summary["high_count"] += 1
            elif severity in ["medium", "moderate"]:
                summary["medium_count"] += 1
            elif severity in ["low"]:
                summary["low_count"] += 1

        self.results.update(
            {
                "scan_type": "dependency",
                "summary": summary,
                "findings": findings,
                "recommendations": self._generate_dependency_recommendations(findings),
            }
        )

        # Generate markdown summary
        self._generate_dependency_summary_md()

        return self.results

    def parse_static_analysis_results(self) -> dict[str, Any]:
        """Parse static code analysis results from Bandit and Semgrep."""
        print("📋 Parsing static analysis results...")

        findings = []
        summary = {
            "total_issues": 0,
            "high_confidence": 0,
            "medium_confidence": 0,
            "low_confidence": 0,
            "tools_used": ["bandit", "semgrep"],
        }

        # Parse Bandit results
        bandit_findings = self._parse_bandit_results()
        findings.extend(bandit_findings)

        # Parse Semgrep results
        semgrep_findings = self._parse_semgrep_results()
        findings.extend(semgrep_findings)

        # Update summary
        for finding in findings:
            confidence = finding.get("confidence", "unknown").lower()
            summary["total_issues"] += 1

            if confidence in ["high"]:
                summary["high_confidence"] += 1
            elif confidence in ["medium"]:
                summary["medium_confidence"] += 1
            elif confidence in ["low"]:
                summary["low_confidence"] += 1

        self.results.update(
            {
                "scan_type": "static_analysis",
                "summary": summary,
                "findings": findings,
                "recommendations": self._generate_static_analysis_recommendations(findings),
            }
        )

        # Generate markdown summary
        self._generate_static_analysis_summary_md()

        return self.results

    def parse_secrets_results(self) -> dict[str, Any]:
        """Parse secrets scanning results."""
        print("📋 Parsing secrets scan results...")

        findings = []
        summary = {
            "total_secrets": 0,
            "high_entropy": 0,
            "potential_keys": 0,
            "false_positives": 0,
            "tools_used": ["detect-secrets", "trufflehog"],
        }

        # Parse detect-secrets results
        detect_secrets_findings = self._parse_detect_secrets_results()
        findings.extend(detect_secrets_findings)

        # Parse TruffleHog results
        trufflehog_findings = self._parse_trufflehog_results()
        findings.extend(trufflehog_findings)

        # Update summary
        for finding in findings:
            summary["total_secrets"] += 1

            if finding.get("entropy", 0) > 4.0:
                summary["high_entropy"] += 1

            if "key" in finding.get("type", "").lower():
                summary["potential_keys"] += 1

        self.results.update(
            {
                "scan_type": "secrets",
                "summary": summary,
                "findings": findings,
                "recommendations": self._generate_secrets_recommendations(findings),
            }
        )

        # Generate markdown summary
        self._generate_secrets_summary_md()

        return self.results

    def parse_container_results(self) -> dict[str, Any]:
        """Parse container security scan results."""
        print("📋 Parsing container scan results...")

        findings = []
        summary = {
            "total_vulnerabilities": 0,
            "dockerfile_issues": 0,
            "critical_count": 0,
            "high_count": 0,
            "medium_count": 0,
            "low_count": 0,
            "tools_used": ["trivy", "hadolint"],
        }

        # Parse Trivy results
        trivy_findings = self._parse_trivy_results()
        findings.extend(trivy_findings)

        # Parse Hadolint results
        hadolint_findings = self._parse_hadolint_results()
        findings.extend(hadolint_findings)
        summary["dockerfile_issues"] = len(hadolint_findings)

        # Update summary counts
        for finding in findings:
            if finding.get("source") == "trivy":
                severity = finding.get("severity", "unknown").lower()
                summary["total_vulnerabilities"] += 1

                if severity == "critical":
                    summary["critical_count"] += 1
                elif severity == "high":
                    summary["high_count"] += 1
                elif severity == "medium":
                    summary["medium_count"] += 1
                elif severity == "low":
                    summary["low_count"] += 1

        self.results.update(
            {
                "scan_type": "container",
                "summary": summary,
                "findings": findings,
                "recommendations": self._generate_container_recommendations(findings),
            }
        )

        # Generate markdown summary
        self._generate_container_summary_md()

        return self.results

    def parse_license_results(self) -> dict[str, Any]:
        """Parse license compliance results."""
        print("📋 Parsing license compliance results...")

        findings = []
        summary = {
            "total_packages": 0,
            "compliant_licenses": 0,
            "non_compliant_licenses": 0,
            "unknown_licenses": 0,
            "tools_used": ["pip-licenses"],
        }

        # Parse pip-licenses results
        license_findings = self._parse_pip_licenses_results()
        findings.extend(license_findings)

        # Update summary
        for finding in findings:
            summary["total_packages"] += 1

            compliance = finding.get("compliance_status", "unknown")
            if compliance == "compliant":
                summary["compliant_licenses"] += 1
            elif compliance == "non_compliant":
                summary["non_compliant_licenses"] += 1
            else:
                summary["unknown_licenses"] += 1

        self.results.update(
            {
                "scan_type": "license",
                "summary": summary,
                "findings": findings,
                "recommendations": self._generate_license_recommendations(findings),
            }
        )

        # Generate markdown summary
        self._generate_license_summary_md()

        return self.results

    def _parse_safety_results(self) -> list[dict[str, Any]]:
        """Parse Safety vulnerability scan results."""
        findings = []
        safety_file = Path("safety-report.json")

        if not safety_file.exists():
            print("⚠️ Safety report not found")
            return findings

        try:
            with open(safety_file) as f:
                safety_data = json.load(f)

            for vuln in safety_data.get("vulnerabilities", []):
                finding = {
                    "source": "safety",
                    "type": "dependency_vulnerability",
                    "package": vuln.get("package_name", "unknown"),
                    "installed_version": vuln.get("analyzed_version", "unknown"),
                    "vulnerability_id": vuln.get("vulnerability_id", "unknown"),
                    "severity": self._map_safety_severity(vuln.get("severity")),
                    "title": vuln.get("advisory", "Unknown vulnerability"),
                    "description": vuln.get("advisory", ""),
                    "fixed_versions": vuln.get("specs", []),
                    "cve": vuln.get("cve", ""),
                    "more_info_url": vuln.get("more_info_url", ""),
                }
                findings.append(finding)

        except (json.JSONDecodeError, KeyError) as e:
            print(f"⚠️ Error parsing Safety results: {e}")

        return findings

    def _parse_pip_audit_results(self) -> list[dict[str, Any]]:
        """Parse pip-audit vulnerability scan results."""
        findings = []
        audit_file = Path("pip-audit-report.json")

        if not audit_file.exists():
            print("⚠️ pip-audit report not found")
            return findings

        try:
            with open(audit_file) as f:
                audit_data = json.load(f)

            for vuln in audit_data.get("dependencies", []):
                for finding_data in vuln.get("vulns", []):
                    finding = {
                        "source": "pip-audit",
                        "type": "dependency_vulnerability",
                        "package": vuln.get("name", "unknown"),
                        "installed_version": vuln.get("version", "unknown"),
                        "vulnerability_id": finding_data.get("id", "unknown"),
                        "severity": finding_data.get("severity", "unknown").lower(),
                        "title": finding_data.get("description", "Unknown vulnerability"),
                        "description": finding_data.get("description", ""),
                        "fixed_version": finding_data.get("fix_versions", []),
                        "aliases": finding_data.get("aliases", []),
                    }
                    findings.append(finding)

        except (json.JSONDecodeError, KeyError) as e:
            print(f"⚠️ Error parsing pip-audit results: {e}")

        return findings

    def _parse_bandit_results(self) -> list[dict[str, Any]]:
        """Parse Bandit security analysis results."""
        findings = []
        bandit_file = Path("bandit-report.json")

        if not bandit_file.exists():
            print("⚠️ Bandit report not found")
            return findings

        try:
            with open(bandit_file) as f:
                bandit_data = json.load(f)

            for issue in bandit_data.get("results", []):
                finding = {
                    "source": "bandit",
                    "type": "static_analysis",
                    "rule_id": issue.get("test_id", "unknown"),
                    "severity": issue.get("issue_severity", "unknown").lower(),
                    "confidence": issue.get("issue_confidence", "unknown").lower(),
                    "title": issue.get("test_name", "Unknown issue"),
                    "description": issue.get("issue_text", ""),
                    "file": issue.get("filename", "unknown"),
                    "line_number": issue.get("line_number", 0),
                    "code": issue.get("code", ""),
                    "more_info": issue.get("more_info", ""),
                }
                findings.append(finding)

        except (json.JSONDecodeError, KeyError) as e:
            print(f"⚠️ Error parsing Bandit results: {e}")

        return findings

    def _parse_semgrep_results(self) -> list[dict[str, Any]]:
        """Parse Semgrep security analysis results."""
        findings = []
        semgrep_file = Path("semgrep-report.json")

        if not semgrep_file.exists():
            print("⚠️ Semgrep report not found")
            return findings

        try:
            with open(semgrep_file) as f:
                semgrep_data = json.load(f)

            for finding_data in semgrep_data.get("results", []):
                finding = {
                    "source": "semgrep",
                    "type": "static_analysis",
                    "rule_id": finding_data.get("check_id", "unknown"),
                    "severity": finding_data.get("severity", "unknown").lower(),
                    "confidence": "high",  # Semgrep findings are generally high confidence
                    "title": finding_data.get("message", "Unknown issue"),
                    "description": finding_data.get("message", ""),
                    "file": finding_data.get("path", "unknown"),
                    "line_number": finding_data.get("start", {}).get("line", 0),
                    "metadata": finding_data.get("metadata", {}),
                }
                findings.append(finding)

        except (json.JSONDecodeError, KeyError) as e:
            print(f"⚠️ Error parsing Semgrep results: {e}")

        return findings

    def _parse_detect_secrets_results(self) -> list[dict[str, Any]]:
        """Parse detect-secrets results."""
        findings = []
        secrets_file = Path(".secrets.baseline")

        if not secrets_file.exists():
            print("⚠️ detect-secrets baseline not found")
            return findings

        try:
            with open(secrets_file) as f:
                secrets_data = json.load(f)

            for file_path, secrets in secrets_data.get("results", {}).items():
                for secret in secrets:
                    finding = {
                        "source": "detect-secrets",
                        "type": "secret",
                        "secret_type": secret.get("type", "unknown"),
                        "file": file_path,
                        "line_number": secret.get("line_number", 0),
                        "hashed_secret": secret.get("hashed_secret", ""),
                        "is_verified": secret.get("is_verified", False),
                    }
                    findings.append(finding)

        except (json.JSONDecodeError, KeyError) as e:
            print(f"⚠️ Error parsing detect-secrets results: {e}")

        return findings

    def _parse_trufflehog_results(self) -> list[dict[str, Any]]:
        """Parse TruffleHog secrets scan results."""
        findings = []
        trufflehog_file = Path("trufflehog-report.json")

        if not trufflehog_file.exists():
            print("⚠️ TruffleHog report not found")
            return findings

        try:
            with open(trufflehog_file) as f:
                for line in f:
                    if line.strip():
                        finding_data = json.loads(line)
                        finding = {
                            "source": "trufflehog",
                            "type": "secret",
                            "detector_name": finding_data.get("DetectorName", "unknown"),
                            "file": finding_data.get("SourceMetadata", {})
                            .get("Data", {})
                            .get("Filesystem", {})
                            .get("file", "unknown"),
                            "line_number": finding_data.get("SourceMetadata", {})
                            .get("Data", {})
                            .get("Filesystem", {})
                            .get("line", 0),
                            "verified": finding_data.get("Verified", False),
                            "raw": finding_data.get("Raw", ""),
                            "entropy": len(finding_data.get("Raw", "")) * 0.1,  # Rough entropy estimate
                        }
                        findings.append(finding)

        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"⚠️ Error parsing TruffleHog results: {e}")

        return findings

    def _parse_trivy_results(self) -> list[dict[str, Any]]:
        """Parse Trivy container vulnerability scan results."""
        findings = []
        trivy_file = Path("trivy-report.json")

        if not trivy_file.exists():
            print("⚠️ Trivy report not found")
            return findings

        try:
            with open(trivy_file) as f:
                trivy_data = json.load(f)

            for result in trivy_data.get("Results", []):
                for vuln in result.get("Vulnerabilities", []):
                    finding = {
                        "source": "trivy",
                        "type": "container_vulnerability",
                        "vulnerability_id": vuln.get("VulnerabilityID", "unknown"),
                        "package": vuln.get("PkgName", "unknown"),
                        "installed_version": vuln.get("InstalledVersion", "unknown"),
                        "fixed_version": vuln.get("FixedVersion", ""),
                        "severity": vuln.get("Severity", "unknown").lower(),
                        "title": vuln.get("Title", "Unknown vulnerability"),
                        "description": vuln.get("Description", ""),
                        "references": vuln.get("References", []),
                    }
                    findings.append(finding)

        except (json.JSONDecodeError, KeyError) as e:
            print(f"⚠️ Error parsing Trivy results: {e}")

        return findings

    def _parse_hadolint_results(self) -> list[dict[str, Any]]:
        """Parse Hadolint Dockerfile analysis results."""
        findings = []
        hadolint_file = Path("hadolint-report.txt")

        if not hadolint_file.exists():
            print("⚠️ Hadolint report not found")
            return findings

        try:
            with open(hadolint_file) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        # Parse Hadolint output format
                        parts = line.split(" ", 2)
                        if len(parts) >= 3:
                            finding = {
                                "source": "hadolint",
                                "type": "dockerfile_issue",
                                "file": "Dockerfile",
                                "line": parts[0] if parts[0].startswith("DL") else "unknown",
                                "severity": "medium",  # Hadolint doesn't provide severity
                                "rule": parts[1] if len(parts) > 1 else "unknown",
                                "description": parts[2] if len(parts) > 2 else line,
                            }
                            findings.append(finding)

        except Exception as e:
            print(f"⚠️ Error parsing Hadolint results: {e}")

        return findings

    def _parse_pip_licenses_results(self) -> list[dict[str, Any]]:
        """Parse pip-licenses compliance results."""
        findings = []
        license_file = Path("license-report.json")

        if not license_file.exists():
            print("⚠️ License report not found")
            return findings

        # Define acceptable licenses for scientific/academic software
        acceptable_licenses = {
            "MIT",
            "BSD",
            "BSD-2-Clause",
            "BSD-3-Clause",
            "Apache",
            "Apache-2.0",
            "Apache Software License",
            "ISC",
            "Python Software Foundation License",
            "PSF",
            "Mozilla Public License 2.0 (MPL 2.0)",
            "MPL-2.0",
        }

        try:
            with open(license_file) as f:
                license_data = json.load(f)

            for package in license_data:
                license_name = package.get("License", "Unknown")
                compliance_status = "compliant" if license_name in acceptable_licenses else "non_compliant"

                if license_name in ["UNKNOWN", "Unknown", ""]:
                    compliance_status = "unknown"

                finding = {
                    "source": "pip-licenses",
                    "type": "license_compliance",
                    "package": package.get("Name", "unknown"),
                    "version": package.get("Version", "unknown"),
                    "license": license_name,
                    "compliance_status": compliance_status,
                    "author": package.get("Author", ""),
                    "description": package.get("Description", ""),
                }
                findings.append(finding)

        except (json.JSONDecodeError, KeyError) as e:
            print(f"⚠️ Error parsing license results: {e}")

        return findings

    def _map_safety_severity(self, severity: str | None) -> str:
        """Map Safety severity levels to standardized format."""
        if not severity:
            return "unknown"

        severity_map = {"70": "critical", "60": "high", "50": "medium", "40": "low", "30": "low"}

        return severity_map.get(str(severity), severity.lower())

    def _generate_dependency_recommendations(self, findings: list[dict[str, Any]]) -> list[str]:
        """Generate recommendations based on dependency scan findings."""
        recommendations = []

        if not findings:
            recommendations.append("✅ No dependency vulnerabilities found")
            return recommendations

        critical_count = sum(1 for f in findings if f.get("severity") == "critical")
        high_count = sum(1 for f in findings if f.get("severity") == "high")

        if critical_count > 0:
            recommendations.append(
                f"🚨 {critical_count} critical vulnerabilities found - update dependencies immediately"
            )

        if high_count > 0:
            recommendations.append(f"⚠️ {high_count} high severity vulnerabilities found - plan updates soon")

        # Group by package for specific recommendations
        packages_with_vulns = {}
        for finding in findings:
            package = finding.get("package", "unknown")
            if package not in packages_with_vulns:
                packages_with_vulns[package] = []
            packages_with_vulns[package].append(finding)

        for package, vulns in list(packages_with_vulns.items())[:5]:  # Top 5 packages
            fixed_versions = []
            for vuln in vulns:
                if vuln.get("fixed_versions"):
                    fixed_versions.extend(vuln["fixed_versions"])

            if fixed_versions:
                recommendations.append(f"📦 Update {package} to version {fixed_versions[0]} or later")

        recommendations.append("🔄 Run 'pip install --upgrade' for affected packages")
        recommendations.append("📋 Consider using dependabot for automated dependency updates")

        return recommendations

    def _generate_static_analysis_recommendations(self, findings: list[dict[str, Any]]) -> list[str]:
        """Generate recommendations based on static analysis findings."""
        recommendations = []

        if not findings:
            recommendations.append("✅ No static analysis issues found")
            return recommendations

        high_confidence = sum(1 for f in findings if f.get("confidence") == "high")

        if high_confidence > 0:
            recommendations.append(f"🔍 {high_confidence} high-confidence security issues found")

        # Group by issue type
        issue_types = {}
        for finding in findings:
            rule_id = finding.get("rule_id", "unknown")
            if rule_id not in issue_types:
                issue_types[rule_id] = 0
            issue_types[rule_id] += 1

        for rule_id, count in sorted(issue_types.items(), key=lambda x: x[1], reverse=True)[:3]:
            recommendations.append(f"🔧 Address {count} instances of {rule_id}")

        recommendations.append("📖 Review Bandit and Semgrep documentation for remediation guidance")
        recommendations.append("🛠️ Consider adding pre-commit hooks for static analysis")

        return recommendations

    def _generate_secrets_recommendations(self, findings: list[dict[str, Any]]) -> list[str]:
        """Generate recommendations based on secrets scan findings."""
        recommendations = []

        if not findings:
            recommendations.append("✅ No secrets detected")
            return recommendations

        verified_secrets = sum(1 for f in findings if f.get("verified", False))

        if verified_secrets > 0:
            recommendations.append(f"🚨 {verified_secrets} verified secrets found - rotate immediately")

        recommendations.append("🔐 Review flagged locations and remove any hardcoded secrets")
        recommendations.append("🔄 Use environment variables or secret management systems")
        recommendations.append("📝 Update .gitignore to exclude sensitive files")
        recommendations.append("🔍 Set up pre-commit hooks for secret detection")

        return recommendations

    def _generate_container_recommendations(self, findings: list[dict[str, Any]]) -> list[str]:
        """Generate recommendations based on container scan findings."""
        recommendations = []

        trivy_findings = [f for f in findings if f.get("source") == "trivy"]
        dockerfile_issues = [f for f in findings if f.get("source") == "hadolint"]

        if not trivy_findings and not dockerfile_issues:
            recommendations.append("✅ No container security issues found")
            return recommendations

        critical_vulns = sum(1 for f in trivy_findings if f.get("severity") == "critical")
        high_vulns = sum(1 for f in trivy_findings if f.get("severity") == "high")

        if critical_vulns > 0:
            recommendations.append(f"🚨 {critical_vulns} critical container vulnerabilities found")

        if high_vulns > 0:
            recommendations.append(f"⚠️ {high_vulns} high severity container vulnerabilities found")

        if dockerfile_issues:
            recommendations.append(f"🐳 {len(dockerfile_issues)} Dockerfile issues found")

        recommendations.append("📦 Update base image to latest secure version")
        recommendations.append("🔄 Regularly scan container images in CI/CD")
        recommendations.append("📋 Follow Dockerfile best practices")

        return recommendations

    def _generate_license_recommendations(self, findings: list[dict[str, Any]]) -> list[str]:
        """Generate recommendations based on license compliance findings."""
        recommendations = []

        non_compliant = [f for f in findings if f.get("compliance_status") == "non_compliant"]
        unknown = [f for f in findings if f.get("compliance_status") == "unknown"]

        if not non_compliant and not unknown:
            recommendations.append("✅ All licenses are compliant")
            return recommendations

        if non_compliant:
            recommendations.append(f"⚠️ {len(non_compliant)} packages with non-compliant licenses")
            for finding in non_compliant[:3]:  # Show first 3
                package = finding.get("package", "unknown")
                license_name = finding.get("license", "unknown")
                recommendations.append(f"📦 {package}: {license_name}")

        if unknown:
            recommendations.append(f"❓ {len(unknown)} packages with unknown licenses")

        recommendations.append("📋 Review license compliance policy")
        recommendations.append("🔍 Consider license scanning in pre-commit hooks")

        return recommendations

    def _generate_dependency_summary_md(self):
        """Generate markdown summary for dependency scan."""
        summary = self.results["summary"]
        findings = self.results["findings"]

        content = f"""# Dependency Security Scan Summary

**Scan Date:** {self.timestamp}
**Tools Used:** {", ".join(summary["tools_used"])}

## Summary
- **Total Vulnerabilities:** {summary["total_vulnerabilities"]}
- **Critical:** {summary["critical_count"]}
- **High:** {summary["high_count"]}
- **Medium:** {summary["medium_count"]}
- **Low:** {summary["low_count"]}

## Recommendations
"""

        for rec in self.results["recommendations"]:
            content += f"- {rec}\n"

        if findings:
            content += "\n## Top Findings\n"
            for finding in findings[:5]:
                content += f"- **{finding.get('package', 'unknown')}**: {finding.get('title', 'Unknown')}\n"

        with open("dependency-summary.md", "w") as f:
            f.write(content)

    def _generate_static_analysis_summary_md(self):
        """Generate markdown summary for static analysis."""
        summary = self.results["summary"]

        content = f"""# Static Analysis Security Summary

**Scan Date:** {self.timestamp}
**Tools Used:** {", ".join(summary["tools_used"])}

## Summary
- **Total Issues:** {summary["total_issues"]}
- **High Confidence:** {summary["high_confidence"]}
- **Medium Confidence:** {summary["medium_confidence"]}
- **Low Confidence:** {summary["low_confidence"]}

## Recommendations
"""

        for rec in self.results["recommendations"]:
            content += f"- {rec}\n"

        with open("static-analysis-summary.md", "w") as f:
            f.write(content)

    def _generate_secrets_summary_md(self):
        """Generate markdown summary for secrets scan."""
        summary = self.results["summary"]

        content = f"""# Secrets Scan Summary

**Scan Date:** {self.timestamp}
**Tools Used:** {", ".join(summary["tools_used"])}

## Summary
- **Total Secrets Found:** {summary["total_secrets"]}
- **High Entropy:** {summary["high_entropy"]}
- **Potential Keys:** {summary["potential_keys"]}

## Recommendations
"""

        for rec in self.results["recommendations"]:
            content += f"- {rec}\n"

        with open("secrets-summary.md", "w") as f:
            f.write(content)

    def _generate_container_summary_md(self):
        """Generate markdown summary for container scan."""
        summary = self.results["summary"]

        content = f"""# Container Security Summary

**Scan Date:** {self.timestamp}
**Tools Used:** {", ".join(summary["tools_used"])}

## Summary
- **Container Vulnerabilities:** {summary["total_vulnerabilities"]}
- **Critical:** {summary["critical_count"]}
- **High:** {summary["high_count"]}
- **Medium:** {summary["medium_count"]}
- **Low:** {summary["low_count"]}
- **Dockerfile Issues:** {summary["dockerfile_issues"]}

## Recommendations
"""

        for rec in self.results["recommendations"]:
            content += f"- {rec}\n"

        with open("container-summary.md", "w") as f:
            f.write(content)

    def _generate_license_summary_md(self):
        """Generate markdown summary for license compliance."""
        summary = self.results["summary"]

        content = f"""# License Compliance Summary

**Scan Date:** {self.timestamp}
**Tools Used:** {", ".join(summary["tools_used"])}

## Summary
- **Total Packages:** {summary["total_packages"]}
- **Compliant Licenses:** {summary["compliant_licenses"]}
- **Non-compliant Licenses:** {summary["non_compliant_licenses"]}
- **Unknown Licenses:** {summary["unknown_licenses"]}

## Recommendations
"""

        for rec in self.results["recommendations"]:
            content += f"- {rec}\n"

        with open("license-summary.md", "w") as f:
            f.write(content)


def main():
    """Main function for parsing security results."""
    parser = argparse.ArgumentParser(description="Parse security scan results")
    parser.add_argument(
        "scan_type",
        choices=["dependency", "static", "secrets", "container", "license"],
        help="Type of security scan to parse",
    )

    args = parser.parse_args()

    results_parser = SecurityResultsParser()

    if args.scan_type == "dependency":
        results = results_parser.parse_dependency_results()
    elif args.scan_type == "static":
        results = results_parser.parse_static_analysis_results()
    elif args.scan_type == "secrets":
        results = results_parser.parse_secrets_results()
    elif args.scan_type == "container":
        results = results_parser.parse_container_results()
    elif args.scan_type == "license":
        results = results_parser.parse_license_results()

    # Save results to JSON file
    output_file = f"{args.scan_type}-security-report.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✅ Security results parsed and saved to {output_file}")

    # Print summary
    summary = results["summary"]
    print(f"\n📊 Summary for {args.scan_type} scan:")
    for key, value in summary.items():
        if key != "tools_used":
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
