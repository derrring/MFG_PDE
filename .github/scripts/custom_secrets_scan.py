#!/usr/bin/env python3
"""
Custom secrets scanning script for MFG_PDE security pipeline.
Scans for hardcoded secrets, API keys, and sensitive information.
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple


class CustomSecretsScanner:
    """Custom scanner for secrets and sensitive information."""
    
    # Patterns for different types of secrets
    SECRET_PATTERNS = {
        "api_key": [
            r"(?i)api[_-]?key['\"\s]*[:=]['\"\s]*[a-zA-Z0-9]{20,}",
            r"(?i)apikey['\"\s]*[:=]['\"\s]*[a-zA-Z0-9]{20,}",
        ],
        "secret_key": [
            r"(?i)secret[_-]?key['\"\s]*[:=]['\"\s]*[a-zA-Z0-9]{20,}",
            r"(?i)secretkey['\"\s]*[:=]['\"\s]*[a-zA-Z0-9]{20,}",
        ],
        "password": [
            r"(?i)password['\"\s]*[:=]['\"\s]*[a-zA-Z0-9!@#$%^&*()_+\-=\[\]{};':\"\\|,.<>\/?]{8,}",
            r"(?i)passwd['\"\s]*[:=]['\"\s]*[a-zA-Z0-9!@#$%^&*()_+\-=\[\]{};':\"\\|,.<>\/?]{8,}",
            r"(?i)pwd['\"\s]*[:=]['\"\s]*[a-zA-Z0-9!@#$%^&*()_+\-=\[\]{};':\"\\|,.<>\/?]{8,}",
        ],
        "token": [
            r"(?i)token['\"\s]*[:=]['\"\s]*[a-zA-Z0-9._\-]{20,}",
            r"(?i)access[_-]?token['\"\s]*[:=]['\"\s]*[a-zA-Z0-9._\-]{20,}",
            r"(?i)auth[_-]?token['\"\s]*[:=]['\"\s]*[a-zA-Z0-9._\-]{20,}",
        ],
        "private_key": [
            r"-----BEGIN [A-Z ]+PRIVATE KEY-----",
            r"-----BEGIN RSA PRIVATE KEY-----",
            r"-----BEGIN DSA PRIVATE KEY-----",
            r"-----BEGIN EC PRIVATE KEY-----",
        ],
        "database_url": [
            r"(?i)(?:postgres|mysql|mongodb)://[^\s\"']+",
            r"(?i)database[_-]?url['\"\s]*[:=]['\"\s]*[a-zA-Z0-9+.:/\-_@]+",
        ],
        "aws_key": [
            r"AKIA[0-9A-Z]{16}",
            r"(?i)aws[_-]?access[_-]?key[_-]?id['\"\s]*[:=]['\"\s]*[A-Z0-9]{20}",
            r"(?i)aws[_-]?secret[_-]?access[_-]?key['\"\s]*[:=]['\"\s]*[A-Za-z0-9/+=]{40}",
        ],
        "github_token": [
            r"ghp_[a-zA-Z0-9]{36}",
            r"gho_[a-zA-Z0-9]{36}",
            r"ghu_[a-zA-Z0-9]{36}",
            r"ghs_[a-zA-Z0-9]{36}",
            r"ghr_[a-zA-Z0-9]{36}",
        ],
        "jwt_token": [
            r"eyJ[a-zA-Z0-9_\-]*\.eyJ[a-zA-Z0-9_\-]*\.[a-zA-Z0-9_\-]*",
        ],
        "email_credentials": [
            r"(?i)smtp[_-]?password['\"\s]*[:=]['\"\s]*[a-zA-Z0-9!@#$%^&*()_+\-=\[\]{};':\"\\|,.<>\/?]{8,}",
            r"(?i)email[_-]?password['\"\s]*[:=]['\"\s]*[a-zA-Z0-9!@#$%^&*()_+\-=\[\]{};':\"\\|,.<>\/?]{8,}",
        ],
    }
    
    # File extensions to scan
    SCAN_EXTENSIONS = {
        ".py", ".js", ".ts", ".jsx", ".tsx", ".yaml", ".yml", 
        ".json", ".env", ".sh", ".bash", ".zsh", ".fish",
        ".toml", ".cfg", ".ini", ".conf", ".config",
        ".md", ".txt", ".rst", ".ipynb"
    }
    
    # Directories to exclude
    EXCLUDE_DIRS = {
        ".git", "__pycache__", ".pytest_cache", ".mypy_cache",
        "node_modules", ".tox", ".venv", "venv", "env",
        "build", "dist", ".eggs", "*.egg-info",
        "archive", ".archive"
    }
    
    # Files to exclude
    EXCLUDE_FILES = {
        ".gitignore", ".gitmodules", ".gitattributes",
        "package-lock.json", "yarn.lock", "poetry.lock",
        "requirements.txt", "setup.py", "setup.cfg"
    }

    def __init__(self, root_dir: str = "."):
        self.root_dir = Path(root_dir)
        self.findings = []
        self.false_positives = set()
        self.load_whitelist()

    def load_whitelist(self):
        """Load whitelist of known false positives."""
        # Common false positives in scientific computing
        self.false_positives = {
            # Mathematical constants and examples
            "example_key", "demo_password", "test_token",
            "your_api_key", "insert_key_here", "replace_with_key",
            "dummy_password", "fake_secret", "placeholder",
            # Documentation examples
            "sk-test", "pk_test", "demo_", "example_",
            # Common variable names
            "api_key_var", "secret_key_var", "token_var",
        }

    def is_whitelisted(self, content: str, match: str) -> bool:
        """Check if a match is a known false positive."""
        content_lower = content.lower()
        match_lower = match.lower()
        
        # Check explicit whitelist
        for fp in self.false_positives:
            if fp in match_lower:
                return True
        
        # Check if it's in comments or documentation
        if any(marker in content_lower for marker in ["#", "//", "/*", "\"\"\"", "'''"]):
            return True
            
        # Check if it's a template or example
        if any(marker in match_lower for marker in ["example", "demo", "test", "placeholder", "dummy"]):
            return True
            
        return False

    def scan_file(self, file_path: Path) -> List[Dict]:
        """Scan a single file for secrets."""
        findings = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            lines = content.splitlines()
            
            for secret_type, patterns in self.SECRET_PATTERNS.items():
                for pattern in patterns:
                    for match in re.finditer(pattern, content, re.MULTILINE):
                        matched_text = match.group(0)
                        
                        # Skip if whitelisted
                        if self.is_whitelisted(content, matched_text):
                            continue
                        
                        # Find line number
                        line_num = content[:match.start()].count('\n') + 1
                        
                        # Get context
                        context_start = max(0, line_num - 2)
                        context_end = min(len(lines), line_num + 1)
                        context = lines[context_start:context_end]
                        
                        finding = {
                            "file": str(file_path.relative_to(self.root_dir)),
                            "line": line_num,
                            "type": secret_type,
                            "pattern": pattern,
                            "match": matched_text[:50] + "..." if len(matched_text) > 50 else matched_text,
                            "severity": self.get_severity(secret_type),
                            "context": context
                        }
                        
                        findings.append(finding)
                        
        except Exception as e:
            print(f"Error scanning {file_path}: {e}")
            
        return findings

    def get_severity(self, secret_type: str) -> str:
        """Get severity level for secret type."""
        high_severity = {"private_key", "aws_key", "database_url", "github_token"}
        medium_severity = {"api_key", "secret_key", "token", "jwt_token"}
        
        if secret_type in high_severity:
            return "HIGH"
        elif secret_type in medium_severity:
            return "MEDIUM"
        else:
            return "LOW"

    def should_scan_file(self, file_path: Path) -> bool:
        """Check if file should be scanned."""
        # Check file extension
        if file_path.suffix not in self.SCAN_EXTENSIONS:
            return False
            
        # Check if file is excluded
        if file_path.name in self.EXCLUDE_FILES:
            return False
            
        # Check if any parent directory is excluded
        for part in file_path.parts:
            if part in self.EXCLUDE_DIRS:
                return False
                
        return True

    def scan_directory(self) -> List[Dict]:
        """Scan entire directory tree for secrets."""
        print(f"🔍 Scanning {self.root_dir} for secrets...")
        
        for file_path in self.root_dir.rglob("*"):
            if file_path.is_file() and self.should_scan_file(file_path):
                file_findings = self.scan_file(file_path)
                self.findings.extend(file_findings)
        
        return self.findings

    def generate_report(self) -> Dict:
        """Generate comprehensive secrets scan report."""
        # Group findings by severity
        by_severity = {"HIGH": [], "MEDIUM": [], "LOW": []}
        by_type = {}
        
        for finding in self.findings:
            severity = finding["severity"]
            secret_type = finding["type"]
            
            by_severity[severity].append(finding)
            
            if secret_type not in by_type:
                by_type[secret_type] = []
            by_type[secret_type].append(finding)
        
        report = {
            "summary": {
                "total_findings": len(self.findings),
                "high_severity": len(by_severity["HIGH"]),
                "medium_severity": len(by_severity["MEDIUM"]),
                "low_severity": len(by_severity["LOW"]),
                "files_scanned": len(set(f["file"] for f in self.findings)),
                "secret_types_found": list(by_type.keys())
            },
            "findings_by_severity": by_severity,
            "findings_by_type": by_type,
            "all_findings": self.findings
        }
        
        return report

    def save_results(self):
        """Save scan results to files."""
        report = self.generate_report()
        
        # Save JSON report
        with open("custom-secrets-report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        # Generate markdown summary
        markdown_report = self.generate_markdown_report(report)
        with open("secrets-summary.md", "w") as f:
            f.write(markdown_report)
        
        print(f"Custom secrets scan completed.")
        print(f"Total findings: {report['summary']['total_findings']}")
        
        if report['summary']['high_severity'] > 0:
            print(f"❌ Found {report['summary']['high_severity']} HIGH severity secrets!")
            sys.exit(1)
        elif report['summary']['medium_severity'] > 0:
            print(f"⚠️ Found {report['summary']['medium_severity']} MEDIUM severity secrets.")
        
        print("✅ No critical secrets found.")

    def generate_markdown_report(self, report: Dict) -> str:
        """Generate markdown report."""
        lines = []
        lines.append("# Custom Secrets Scan Report")
        lines.append("")
        
        summary = report["summary"]
        lines.append("## Summary")
        lines.append(f"- **Total Findings**: {summary['total_findings']}")
        lines.append(f"- **High Severity**: {summary['high_severity']}")
        lines.append(f"- **Medium Severity**: {summary['medium_severity']}")
        lines.append(f"- **Low Severity**: {summary['low_severity']}")
        lines.append(f"- **Files Scanned**: {summary['files_scanned']}")
        lines.append("")
        
        if summary["high_severity"] > 0:
            lines.append("## ❌ High Severity Findings")
            for finding in report["findings_by_severity"]["HIGH"]:
                lines.append(f"- **{finding['file']}:{finding['line']}** - {finding['type']}")
                lines.append(f"  - Match: `{finding['match']}`")
            lines.append("")
        
        if summary["medium_severity"] > 0:
            lines.append("## ⚠️ Medium Severity Findings")
            for finding in report["findings_by_severity"]["MEDIUM"]:
                lines.append(f"- **{finding['file']}:{finding['line']}** - {finding['type']}")
                lines.append(f"  - Match: `{finding['match']}`")
            lines.append("")
        
        return "\n".join(lines)


def main():
    """Main entry point."""
    scanner = CustomSecretsScanner()
    findings = scanner.scan_directory()
    scanner.save_results()


if __name__ == "__main__":
    main()