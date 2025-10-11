# Dynamic Coverage Badge Setup

This guide explains how to set up the dynamic test coverage badge using GitHub Gists.

## Prerequisites

- GitHub repository with admin access
- GitHub personal access token with gist permissions

## Setup Steps

### 1. Create GitHub Personal Access Token

1. Go to GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Click "Generate new token (classic)"
3. Name: "Coverage Badge Token"
4. Select scope: **`gist`** (only this permission needed)
5. Click "Generate token"
6. **Copy the token** (you won't see it again!)

### 2. Create a Gist for Badge Data

1. Go to https://gist.github.com/
2. Create a new gist:
   - Filename: `coverage.json`
   - Content: `{"schemaVersion": 1, "label": "coverage", "message": "unknown", "color": "lightgrey"}`
3. Make it **public**
4. Click "Create public gist"
5. **Copy the gist ID** from the URL (e.g., `https://gist.github.com/username/abc123def456` → `abc123def456`)

### 3. Add Secrets to GitHub Repository

1. Go to repository Settings → Secrets and variables → Actions
2. Add two secrets:

**Secret 1: GIST_SECRET**
- Click "New repository secret"
- Name: `GIST_SECRET`
- Value: Paste the personal access token from step 1
- Click "Add secret"

### 4. Update CI Workflow

The gist ID in `.github/workflows/ci.yml` should match your gist:

```yaml
- name: Create coverage badge
  uses: schneegans/dynamic-badges-action@v1.7.0
  with:
    auth: ${{ secrets.GIST_SECRET }}
    gistID: abc123def456  # Replace with your gist ID
    filename: coverage.json
```

### 5. Verify Setup

Once configured:

1. Push a commit to trigger CI workflow
2. Check GitHub Actions tab for workflow run
3. Test coverage job should complete successfully
4. Badge in README.md will update with actual coverage percentage
5. Check your gist - it should now contain coverage data

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
