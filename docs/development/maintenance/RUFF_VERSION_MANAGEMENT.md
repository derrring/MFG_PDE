# Ruff Version Management

**Strategy**: Pin with Periodic Automated Updates

## Overview

MFG_PDE pins the ruff version to ensure consistent formatting across all environments (local dev, pre-commit hooks, CI/CD). This provides:

- ✅ **Consistency**: Same formatting behavior everywhere
- ✅ **Reproducibility**: Code formatting is deterministic
- ✅ **Stability**: No surprise CI failures from ruff updates
- ✅ **Controlled evolution**: Review formatting changes explicitly

## Current Version

```yaml
# .pre-commit-config.yaml
rev: v0.13.1

# .github/workflows/modern_quality.yml
pip install ruff==0.13.1
```

**Last Updated**: 2025-10-07

## Automated Update System

### Monthly Automated Check (Recommended)

A GitHub Action runs automatically on the **1st of each month** to check for ruff updates:

- **Workflow**: `.github/workflows/check-ruff-updates.yml`
- **Schedule**: First day of month at 9 AM UTC
- **Behavior**:
  - Checks for latest stable ruff release
  - If update available, creates PR with:
    - Updated configuration files
    - Applied formatting changes
    - Link to ruff release notes
  - Labels PR with: `dependencies`, `tooling`, `automated`

**Manual Trigger**:
```bash
# Via GitHub Actions UI:
# Actions → Check Ruff Version Updates → Run workflow
```

### Manual Update Script (As Needed)

For immediate updates or checking between scheduled runs:

```bash
# Check if update is available
python scripts/update_ruff_version.py --check

# Apply update interactively
python scripts/update_ruff_version.py --update

# Force specific version
python scripts/update_ruff_version.py --force 0.14.0
```

**What it does**:
1. Fetches latest ruff version from GitHub
2. Compares with current version
3. Updates both `.pre-commit-config.yaml` and CI workflow
4. Runs `ruff format` on codebase
5. Shows next steps (review, test, commit)

## Update Review Process

When a ruff update PR is created (automated or manual):

### 1. Review Formatting Changes
```bash
git diff
```
- Check if formatting changes are reasonable
- Look for any unexpected reformatting
- Ensure no semantic changes (just style)

### 2. Run Tests Locally
```bash
pytest tests/
```
- Ensure all tests still pass
- Check for any test failures related to formatting

### 3. Run Pre-commit Hooks
```bash
pre-commit run --all-files
```
- Verify pre-commit hooks work with new version
- Check that all files pass formatting checks

### 4. Merge Decision

**Merge if**:
- ✅ Formatting changes look good
- ✅ All tests pass
- ✅ Pre-commit hooks pass
- ✅ No breaking changes in ruff release notes

**Defer if**:
- ⚠️ Large formatting changes need discussion
- ⚠️ Breaking changes in ruff functionality
- ⚠️ Test failures or unexpected behavior

## Update Frequency Recommendations

**Stable Updates (Recommended)**:
- **Frequency**: Monthly automated check (1st of month)
- **Version type**: Stable releases only (not pre-releases)
- **Review**: Standard review process

**Fast Track Updates**:
- **Security fixes**: Apply immediately using manual script
- **Critical bug fixes**: Apply within 1 week
- **New features**: Follow standard monthly schedule

**Deferred Updates**:
- **Breaking changes**: Defer until time for review
- **Major versions**: Schedule dedicated review session
- **Beta/RC versions**: Skip (wait for stable)

## Rollback Procedure

If a ruff update causes issues:

```bash
# 1. Revert the PR
git revert <commit-hash>

# 2. Or manually restore previous version
python scripts/update_ruff_version.py --force 0.13.1

# 3. Commit and push
git add -A
git commit -m "chore: Revert ruff to v0.13.1 due to issues"
git push
```

## Configuration Files

### Files Updated by Automation

1. **`.pre-commit-config.yaml`**
   ```yaml
   - repo: https://github.com/astral-sh/ruff-pre-commit
     rev: v0.13.1  # ← Updated here
   ```

2. **`.github/workflows/modern_quality.yml`**
   ```yaml
   pip install ruff==0.13.1  # ← Updated here
   ```

### Why Both Files?

- **Pre-commit**: Controls local dev and git hooks
- **CI Workflow**: Controls GitHub Actions checks
- **Must match**: Ensures consistent behavior everywhere

## Troubleshooting

### Issue: Update PR has test failures

**Diagnosis**: New ruff version changed formatting rules

**Solution**:
1. Check ruff release notes for breaking changes
2. Run `ruff check --diff mfg_pde/` to see specific changes
3. Evaluate if changes are acceptable
4. If not, defer update and wait for next version

### Issue: Pre-commit hooks fail after update

**Diagnosis**: Version mismatch or configuration issue

**Solution**:
```bash
# Re-install pre-commit hooks
pre-commit uninstall
pre-commit install

# Update pre-commit cache
pre-commit autoupdate

# Try again
pre-commit run --all-files
```

### Issue: CI passes locally but fails on GitHub

**Diagnosis**: Possible version mismatch or caching issue

**Solution**:
```bash
# Check local ruff version
ruff --version

# Should match .pre-commit-config.yaml
grep -A 1 'ruff-pre-commit' .pre-commit-config.yaml

# Force clear pre-commit cache
pre-commit clean
pre-commit install
```

## Version History

| Date | Version | Notes |
|:-----|:--------|:------|
| 2025-10-07 | 0.13.1 | Initial pin with automated update system |

## References

- **Ruff Releases**: https://github.com/astral-sh/ruff/releases
- **Ruff Changelog**: https://github.com/astral-sh/ruff/blob/main/CHANGELOG.md
- **Pre-commit Hooks**: https://pre-commit.com/

---

**Last Updated**: 2025-10-07
**Automation Status**: ✅ Active (monthly checks enabled)
**Next Scheduled Check**: 2025-11-01
