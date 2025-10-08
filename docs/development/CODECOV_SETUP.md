# Codecov Integration Setup

This guide explains how to set up Codecov for dynamic test coverage badges.

## Prerequisites

- GitHub repository with admin access
- Codecov account (free for public repositories)

## Setup Steps

### 1. Create Codecov Account

1. Go to https://codecov.io/
2. Sign in with your GitHub account
3. Authorize Codecov to access your repositories

### 2. Add Repository to Codecov

1. In Codecov dashboard, click "Add new repository"
2. Find and select `derrring/MFG_PDE`
3. Codecov will provide a repository upload token

### 3. Add Token to GitHub Secrets

1. Go to repository Settings → Secrets and variables → Actions
2. Click "New repository secret"
3. Name: `CODECOV_TOKEN`
4. Value: Paste the token from Codecov dashboard
5. Click "Add secret"

### 4. Verify Setup

Once the token is added:

1. Push a commit to trigger CI workflow
2. Check GitHub Actions tab for workflow run
3. Test coverage job should complete successfully
4. Coverage report will appear on Codecov dashboard
5. Badge in README.md will update automatically

## Configuration

### Codecov Settings (`codecov.yml`)

```yaml
coverage:
  precision: 2
  round: down
  range: "70...100"

  status:
    project:
      default:
        target: 80%        # Project-wide coverage target
        threshold: 2%      # Allow 2% decrease
    patch:
      default:
        target: 70%        # New code coverage target
        threshold: 5%      # Allow 5% variance for patches
```

### Coverage Targets

- **Project Coverage**: 80% (minimum 78% to pass)
- **Patch Coverage**: 70% (for new code in PRs)
- **Color Coding**:
  - Green: 90%+
  - Yellow: 70-90%
  - Red: <70%

## CI Integration

The test coverage job runs on every push and PR:

```yaml
- Run tests with coverage:
    pytest tests/ --cov=mfg_pde --cov-report=xml

- Upload to Codecov:
    uses: codecov/codecov-action@v4
```

Coverage reports are generated in XML format and uploaded automatically.

## Badge Usage

The dynamic badge in README.md:

```markdown
[![codecov](https://codecov.io/gh/derrring/MFG_PDE/branch/main/graph/badge.svg)](https://codecov.io/gh/derrring/MFG_PDE)
```

Features:
- Updates automatically after each test run
- Shows current coverage percentage
- Links to detailed coverage report
- Color-coded by coverage level

## Troubleshooting

### Badge shows "unknown"
- Check that CODECOV_TOKEN is set correctly
- Verify CI workflow completed successfully
- Check Codecov dashboard for upload errors

### Coverage not updating
- Clear browser cache
- Check Codecov dashboard for latest report
- Verify workflow included coverage upload step

### Low coverage warnings
- Review `codecov.yml` thresholds
- Check coverage report for uncovered files
- Add tests for critical uncovered code

## Benefits

1. **Real-time Coverage**: Badge updates automatically
2. **PR Integration**: See coverage impact in pull requests
3. **Trend Analysis**: Track coverage changes over time
4. **File-level Reports**: Identify specific uncovered code
5. **Professional Presentation**: Dynamic badges show active maintenance

## Further Reading

- [Codecov Documentation](https://docs.codecov.com/)
- [GitHub Actions Integration](https://docs.codecov.com/docs/github-actions)
- [Coverage Configuration](https://docs.codecov.com/docs/codecov-yaml)
