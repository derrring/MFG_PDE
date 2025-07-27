#!/usr/bin/env python3
"""
License compliance checking script for MFG_PDE security pipeline.
Checks dependencies for license compatibility and compliance.
"""

import json
import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Set


class LicenseComplianceChecker:
    """Check license compliance for project dependencies."""
    
    # Approved licenses for research/academic use
    APPROVED_LICENSES = {
        "MIT License",
        "MIT",
        "BSD License",
        "BSD",
        "BSD-3-Clause",
        "BSD-2-Clause", 
        "Apache Software License",
        "Apache 2.0",
        "Apache-2.0",
        "Python Software Foundation License",
        "PSF",
        "Mozilla Public License 2.0 (MPL 2.0)",
        "MPL-2.0",
        "ISC License (ISCL)",
        "ISC",
        "GNU Lesser General Public License v3 (LGPLv3)",
        "LGPL-3.0",
        "GNU Lesser General Public License v2.1 (LGPLv2.1)",
        "LGPL-2.1",
    }
    
    # Licenses requiring review
    REVIEW_REQUIRED = {
        "GNU General Public License v3 (GPLv3)",
        "GPL-3.0",
        "GNU General Public License v2 (GPLv2)",
        "GPL-2.0",
        "Copyleft",
    }
    
    # Prohibited licenses
    PROHIBITED_LICENSES = {
        "UNKNOWN",
        "AGPL",
        "Proprietary",
        "Commercial",
    }

    def __init__(self):
        self.results = {
            "approved": [],
            "review_required": [],
            "prohibited": [],
            "unknown": [],
            "summary": {}
        }

    def run_license_check(self) -> Dict:
        """Run pip-licenses and parse results."""
        try:
            result = subprocess.run(
                ["pip-licenses", "--format=json"],
                capture_output=True,
                text=True,
                check=True
            )
            return json.loads(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Error running pip-licenses: {e}")
            return []
        except json.JSONDecodeError as e:
            print(f"Error parsing license data: {e}")
            return []

    def categorize_license(self, license_name: str) -> str:
        """Categorize a license based on our policies."""
        if not license_name or license_name.strip() == "":
            return "unknown"
        
        # Normalize license name
        license_clean = license_name.strip()
        
        if license_clean in self.APPROVED_LICENSES:
            return "approved"
        elif license_clean in self.REVIEW_REQUIRED:
            return "review_required"
        elif license_clean in self.PROHIBITED_LICENSES:
            return "prohibited"
        else:
            # Check for partial matches
            for approved in self.APPROVED_LICENSES:
                if approved.lower() in license_clean.lower():
                    return "approved"
            
            for review in self.REVIEW_REQUIRED:
                if review.lower() in license_clean.lower():
                    return "review_required"
                    
            for prohibited in self.PROHIBITED_LICENSES:
                if prohibited.lower() in license_clean.lower():
                    return "prohibited"
                    
            return "unknown"

    def check_compliance(self) -> Dict:
        """Check license compliance for all dependencies."""
        license_data = self.run_license_check()
        
        for package in license_data:
            name = package.get("Name", "Unknown")
            version = package.get("Version", "Unknown")
            license_name = package.get("License", "UNKNOWN")
            
            category = self.categorize_license(license_name)
            
            package_info = {
                "name": name,
                "version": version,
                "license": license_name,
                "category": category
            }
            
            self.results[category].append(package_info)
        
        # Generate summary
        self.results["summary"] = {
            "total_packages": len(license_data),
            "approved": len(self.results["approved"]),
            "review_required": len(self.results["review_required"]),
            "prohibited": len(self.results["prohibited"]),
            "unknown": len(self.results["unknown"]),
            "compliance_status": "PASS" if len(self.results["prohibited"]) == 0 else "FAIL"
        }
        
        return self.results

    def generate_report(self) -> str:
        """Generate a human-readable compliance report."""
        report = []
        report.append("# License Compliance Report")
        report.append("")
        
        summary = self.results["summary"]
        report.append("## Summary")
        report.append(f"- **Total Packages**: {summary['total_packages']}")
        report.append(f"- **Approved Licenses**: {summary['approved']}")
        report.append(f"- **Review Required**: {summary['review_required']}")
        report.append(f"- **Prohibited**: {summary['prohibited']}")
        report.append(f"- **Unknown**: {summary['unknown']}")
        report.append(f"- **Status**: {summary['compliance_status']}")
        report.append("")
        
        if self.results["prohibited"]:
            report.append("## ❌ Prohibited Licenses")
            for pkg in self.results["prohibited"]:
                report.append(f"- **{pkg['name']}** v{pkg['version']}: {pkg['license']}")
            report.append("")
        
        if self.results["review_required"]:
            report.append("## ⚠️ Licenses Requiring Review")
            for pkg in self.results["review_required"]:
                report.append(f"- **{pkg['name']}** v{pkg['version']}: {pkg['license']}")
            report.append("")
        
        if self.results["unknown"]:
            report.append("## ❓ Unknown Licenses")
            for pkg in self.results["unknown"]:
                report.append(f"- **{pkg['name']}** v{pkg['version']}: {pkg['license']}")
            report.append("")
        
        return "\n".join(report)

    def save_results(self):
        """Save compliance results to files."""
        # Save JSON results
        with open("license-compliance-report.json", "w") as f:
            json.dump(self.results, f, indent=2)
        
        # Save markdown report
        with open("license-summary.md", "w") as f:
            f.write(self.generate_report())
        
        print("License compliance check completed.")
        print(f"Status: {self.results['summary']['compliance_status']}")
        
        if self.results["prohibited"]:
            print(f"❌ Found {len(self.results['prohibited'])} prohibited licenses!")
            sys.exit(1)
        elif self.results["review_required"]:
            print(f"⚠️ Found {len(self.results['review_required'])} licenses requiring review.")
        
        print("✅ License compliance check passed.")


def main():
    """Main entry point."""
    checker = LicenseComplianceChecker()
    results = checker.check_compliance()
    checker.save_results()


if __name__ == "__main__":
    main()