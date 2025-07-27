#!/usr/bin/env python3
"""
Security report generation script for MFG_PDE security pipeline.
Aggregates all security scan results into comprehensive reports.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional


class SecurityReportGenerator:
    """Generate comprehensive security reports from scan results."""
    
    def __init__(self):
        self.scan_results = {}
        self.summary = {
            "timestamp": datetime.now().isoformat(),
            "total_issues": 0,
            "critical_issues": 0,
            "high_issues": 0,
            "medium_issues": 0,
            "low_issues": 0,
            "scans_completed": [],
            "scans_failed": [],
            "overall_status": "UNKNOWN"
        }

    def load_scan_results(self):
        """Load all available scan result files."""
        result_files = {
            "dependency": {
                "files": ["safety-report.json", "pip-audit-report.json"],
                "summary": "dependency-summary.md"
            },
            "static_analysis": {
                "files": ["bandit-report.json", "semgrep-report.json", "custom-security-report.json"],
                "summary": "static-analysis-summary.md"
            },
            "secrets": {
                "files": ["trufflehog-report.json", "custom-secrets-report.json"],
                "summary": "secrets-summary.md"
            },
            "container": {
                "files": ["trivy-report.json", "container-security-report.json"],
                "summary": "container-summary.md"
            },
            "license": {
                "files": ["license-report.json", "license-compliance-report.json"],
                "summary": "license-summary.md"
            }
        }
        
        for scan_type, config in result_files.items():
            self.scan_results[scan_type] = {
                "data": [],
                "summary_exists": False,
                "files_found": []
            }
            
            # Load JSON data files
            for filename in config["files"]:
                if os.path.exists(filename):
                    try:
                        with open(filename, 'r') as f:
                            data = json.load(f)
                            self.scan_results[scan_type]["data"].append({
                                "source": filename,
                                "data": data
                            })
                            self.scan_results[scan_type]["files_found"].append(filename)
                    except Exception as e:
                        print(f"Warning: Could not load {filename}: {e}")
            
            # Check for summary file
            if os.path.exists(config["summary"]):
                self.scan_results[scan_type]["summary_exists"] = True
            
            # Mark scan as completed if any files found
            if self.scan_results[scan_type]["files_found"]:
                self.summary["scans_completed"].append(scan_type)
            else:
                self.summary["scans_failed"].append(scan_type)

    def analyze_dependency_results(self) -> Dict:
        """Analyze dependency scan results."""
        analysis = {
            "vulnerabilities": [],
            "total_packages": 0,
            "vulnerable_packages": 0,
            "severity_counts": {"critical": 0, "high": 0, "medium": 0, "low": 0}
        }
        
        for result in self.scan_results["dependency"]["data"]:
            source = result["source"]
            data = result["data"]
            
            if source == "safety-report.json" and isinstance(data, list):
                for vuln in data:
                    severity = self.map_severity(vuln.get("vulnerability_id", ""), "dependency")
                    analysis["vulnerabilities"].append({
                        "source": "safety",
                        "package": vuln.get("package_name", "Unknown"),
                        "vulnerability_id": vuln.get("vulnerability_id", ""),
                        "severity": severity,
                        "description": vuln.get("advisory", "")
                    })
                    analysis["severity_counts"][severity] += 1
            
            elif source == "pip-audit-report.json" and isinstance(data, dict):
                vulns = data.get("vulnerabilities", [])
                for vuln in vulns:
                    severity = self.map_severity(vuln.get("id", ""), "dependency")
                    analysis["vulnerabilities"].append({
                        "source": "pip-audit",
                        "package": vuln.get("package", "Unknown"),
                        "vulnerability_id": vuln.get("id", ""),
                        "severity": severity,
                        "description": vuln.get("description", "")
                    })
                    analysis["severity_counts"][severity] += 1
        
        analysis["vulnerable_packages"] = len(set(v["package"] for v in analysis["vulnerabilities"]))
        return analysis

    def analyze_static_analysis_results(self) -> Dict:
        """Analyze static analysis results."""
        analysis = {
            "issues": [],
            "total_issues": 0,
            "severity_counts": {"critical": 0, "high": 0, "medium": 0, "low": 0}
        }
        
        for result in self.scan_results["static_analysis"]["data"]:
            source = result["source"]
            data = result["data"]
            
            if source == "bandit-report.json":
                results = data.get("results", [])
                for issue in results:
                    severity = self.map_bandit_severity(issue.get("issue_severity", "LOW"))
                    analysis["issues"].append({
                        "source": "bandit",
                        "file": issue.get("filename", "Unknown"),
                        "line": issue.get("line_number", 0),
                        "severity": severity,
                        "type": issue.get("test_name", ""),
                        "description": issue.get("issue_text", "")
                    })
                    analysis["severity_counts"][severity] += 1
            
            elif source == "semgrep-report.json":
                results = data.get("results", [])
                for issue in results:
                    severity = self.map_semgrep_severity(issue.get("extra", {}).get("severity", "INFO"))
                    analysis["issues"].append({
                        "source": "semgrep",
                        "file": issue.get("path", "Unknown"),
                        "line": issue.get("start", {}).get("line", 0),
                        "severity": severity,
                        "type": issue.get("check_id", ""),
                        "description": issue.get("extra", {}).get("message", "")
                    })
                    analysis["severity_counts"][severity] += 1
            
            elif source == "custom-security-report.json":
                issues = data.get("issues", [])
                for issue in issues:
                    severity = issue.get("severity", "low").lower()
                    analysis["issues"].append({
                        "source": "custom",
                        "file": issue.get("file", "Unknown"),
                        "line": issue.get("line", 0),
                        "severity": severity,
                        "type": issue.get("type", ""),
                        "description": issue.get("description", "")
                    })
                    analysis["severity_counts"][severity] += 1
        
        analysis["total_issues"] = len(analysis["issues"])
        return analysis

    def analyze_secrets_results(self) -> Dict:
        """Analyze secrets scan results."""
        analysis = {
            "secrets": [],
            "total_secrets": 0,
            "severity_counts": {"critical": 0, "high": 0, "medium": 0, "low": 0}
        }
        
        for result in self.scan_results["secrets"]["data"]:
            source = result["source"]
            data = result["data"]
            
            if source == "custom-secrets-report.json":
                findings = data.get("all_findings", [])
                for finding in findings:
                    severity = finding.get("severity", "LOW").lower()
                    analysis["secrets"].append({
                        "source": "custom",
                        "file": finding.get("file", "Unknown"),
                        "line": finding.get("line", 0),
                        "severity": severity,
                        "type": finding.get("type", ""),
                        "match": finding.get("match", "")
                    })
                    analysis["severity_counts"][severity] += 1
        
        analysis["total_secrets"] = len(analysis["secrets"])
        return analysis

    def map_severity(self, vuln_id: str, scan_type: str) -> str:
        """Map vulnerability severity based on ID and type."""
        # Simple heuristic mapping - in practice, use CVE scores
        if any(indicator in vuln_id.upper() for indicator in ["CVE-", "CRITICAL", "HIGH"]):
            return "high"
        elif any(indicator in vuln_id.upper() for indicator in ["MEDIUM", "MODERATE"]):
            return "medium"
        else:
            return "low"

    def map_bandit_severity(self, severity: str) -> str:
        """Map Bandit severity to standard levels."""
        mapping = {
            "HIGH": "high",
            "MEDIUM": "medium", 
            "LOW": "low"
        }
        return mapping.get(severity.upper(), "low")

    def map_semgrep_severity(self, severity: str) -> str:
        """Map Semgrep severity to standard levels."""
        mapping = {
            "ERROR": "high",
            "WARNING": "medium",
            "INFO": "low"
        }
        return mapping.get(severity.upper(), "low")

    def calculate_overall_summary(self):
        """Calculate overall security summary."""
        dependency_analysis = self.analyze_dependency_results()
        static_analysis = self.analyze_static_analysis_results()
        secrets_analysis = self.analyze_secrets_results()
        
        # Aggregate counts
        self.summary["total_issues"] = (
            len(dependency_analysis["vulnerabilities"]) +
            static_analysis["total_issues"] +
            secrets_analysis["total_secrets"]
        )
        
        # Aggregate severity counts
        for analysis in [dependency_analysis, static_analysis, secrets_analysis]:
            severity_counts = analysis.get("severity_counts", {})
            self.summary["critical_issues"] += severity_counts.get("critical", 0)
            self.summary["high_issues"] += severity_counts.get("high", 0)
            self.summary["medium_issues"] += severity_counts.get("medium", 0)
            self.summary["low_issues"] += severity_counts.get("low", 0)
        
        # Determine overall status
        if self.summary["critical_issues"] > 0:
            self.summary["overall_status"] = "CRITICAL"
        elif self.summary["high_issues"] > 0:
            self.summary["overall_status"] = "HIGH"
        elif self.summary["medium_issues"] > 0:
            self.summary["overall_status"] = "MEDIUM"
        elif self.summary["low_issues"] > 0:
            self.summary["overall_status"] = "LOW"
        else:
            self.summary["overall_status"] = "PASS"

    def generate_html_dashboard(self) -> str:
        """Generate HTML security dashboard."""
        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MFG_PDE Security Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .status-badge {{ padding: 5px 15px; border-radius: 20px; color: white; font-weight: bold; }}
        .status-pass {{ background: #28a745; }}
        .status-low {{ background: #ffc107; color: black; }}
        .status-medium {{ background: #fd7e14; }}
        .status-high {{ background: #dc3545; }}
        .status-critical {{ background: #6f42c1; }}
        .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }}
        .summary-card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #007bff; }}
        .scan-results {{ margin: 20px 0; }}
        .scan-section {{ margin: 20px 0; padding: 15px; border: 1px solid #dee2e6; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔒 MFG_PDE Security Dashboard</h1>
            <span class="status-badge status-{self.summary['overall_status'].lower()}">{self.summary['overall_status']}</span>
            <p>Generated: {self.summary['timestamp']}</p>
        </div>
        
        <div class="summary-grid">
            <div class="summary-card">
                <h3>Total Issues</h3>
                <p style="font-size: 2em; margin: 0;">{self.summary['total_issues']}</p>
            </div>
            <div class="summary-card">
                <h3>Critical Issues</h3>
                <p style="font-size: 2em; margin: 0; color: #6f42c1;">{self.summary['critical_issues']}</p>
            </div>
            <div class="summary-card">
                <h3>High Issues</h3>
                <p style="font-size: 2em; margin: 0; color: #dc3545;">{self.summary['high_issues']}</p>
            </div>
            <div class="summary-card">
                <h3>Medium Issues</h3>
                <p style="font-size: 2em; margin: 0; color: #fd7e14;">{self.summary['medium_issues']}</p>
            </div>
        </div>
        
        <div class="scan-results">
            <h2>Scan Results Summary</h2>
            <div class="scan-section">
                <h3>✅ Completed Scans</h3>
                <ul>
                    {"".join(f"<li>{scan}</li>" for scan in self.summary['scans_completed'])}
                </ul>
            </div>
            
            {f'''<div class="scan-section">
                <h3>❌ Failed Scans</h3>
                <ul>
                    {"".join(f"<li>{scan}</li>" for scan in self.summary['scans_failed'])}
                </ul>
            </div>''' if self.summary['scans_failed'] else ''}
        </div>
    </div>
</body>
</html>
        """
        return html_template

    def generate_markdown_summary(self) -> str:
        """Generate markdown security summary."""
        lines = []
        lines.append("# MFG_PDE Security Summary")
        lines.append("")
        lines.append(f"**Generated**: {self.summary['timestamp']}")
        lines.append(f"**Overall Status**: {self.summary['overall_status']}")
        lines.append("")
        
        lines.append("## Summary")
        lines.append(f"- **Total Issues**: {self.summary['total_issues']}")
        lines.append(f"- **Critical**: {self.summary['critical_issues']}")
        lines.append(f"- **High**: {self.summary['high_issues']}")
        lines.append(f"- **Medium**: {self.summary['medium_issues']}")
        lines.append(f"- **Low**: {self.summary['low_issues']}")
        lines.append("")
        
        lines.append("## Scan Status")
        lines.append(f"- **Completed**: {', '.join(self.summary['scans_completed'])}")
        if self.summary['scans_failed']:
            lines.append(f"- **Failed**: {', '.join(self.summary['scans_failed'])}")
        lines.append("")
        
        return "\n".join(lines)

    def save_reports(self):
        """Save all security reports."""
        # Generate comprehensive findings JSON
        findings = {
            "summary": self.summary,
            "dependency_analysis": self.analyze_dependency_results(),
            "static_analysis": self.analyze_static_analysis_results(),
            "secrets_analysis": self.analyze_secrets_results(),
            "scan_results": self.scan_results
        }
        
        with open("security-findings.json", "w") as f:
            json.dump(findings, f, indent=2)
        
        # Generate HTML dashboard
        html_dashboard = self.generate_html_dashboard()
        with open("security-dashboard.html", "w") as f:
            f.write(html_dashboard)
        
        # Generate markdown summary
        markdown_summary = self.generate_markdown_summary()
        with open("security-summary.md", "w") as f:
            f.write(markdown_summary)
        
        print("📊 Security reports generated:")
        print("  - security-findings.json")
        print("  - security-dashboard.html")
        print("  - security-summary.md")
        
        print(f"\n🔒 Security Status: {self.summary['overall_status']}")
        print(f"📊 Total Issues: {self.summary['total_issues']}")
        
        if self.summary["critical_issues"] > 0 or self.summary["high_issues"] > 0:
            print(f"⚠️ Found {self.summary['critical_issues']} critical and {self.summary['high_issues']} high severity issues!")

    def run(self):
        """Run the complete report generation process."""
        print("📊 Generating comprehensive security report...")
        self.load_scan_results()
        self.calculate_overall_summary()
        self.save_reports()


def main():
    """Main entry point."""
    generator = SecurityReportGenerator()
    generator.run()


if __name__ == "__main__":
    main()