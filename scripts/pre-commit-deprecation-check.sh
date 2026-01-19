#!/bin/bash
# Pre-commit hook: Check for internal usage of deprecated APIs
#
# This hook prevents commits that contain production code calling
# deprecated functions. This enforces the deprecation lifecycle policy.
#
# To install: ln -s ../../scripts/pre-commit-deprecation-check.sh .git/hooks/pre-commit
#
# To bypass (emergency only): git commit --no-verify

set -e

echo "Running deprecation check..."

python scripts/check_internal_deprecation.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Commit rejected: Production code uses deprecated APIs"
    echo "   Fix violations or update to new API before committing"
    echo ""
    echo "   To bypass (not recommended): git commit --no-verify"
    exit 1
fi

echo "✅ Deprecation check passed"
exit 0
