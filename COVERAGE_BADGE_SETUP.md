# Coverage Badge Setup Instructions

Quick reference for activating the dynamic coverage badge in README.md.

## Quick Setup (4 Steps)

### 1️⃣ Create GitHub Personal Access Token

Visit: https://github.com/settings/tokens/new

- **Note**: "Coverage Badge Token"
- **Expiration**: No expiration (or 1 year)
- **Scopes**: ✅ `gist` (only this one)
- Click **"Generate token"**
- ⚠️ **COPY THE TOKEN** (you won't see it again!)

### 2️⃣ Create Public Gist

Visit: https://gist.github.com/

- **Description**: "MFG_PDE Coverage Badge"
- **Filename**: `coverage.json`
- **Content**:
  ```json
  {"schemaVersion": 1, "label": "coverage", "message": "unknown", "color": "lightgrey"}
  ```
- Select **"Create public gist"**
- ⚠️ **COPY THE GIST ID** from URL
  - Example: `https://gist.github.com/derrring/abc123def456`
  - Gist ID: `abc123def456`

### 3️⃣ Add Repository Secret

Visit: https://github.com/derrring/MFG_PDE/settings/secrets/actions

- Click **"New repository secret"**
- **Name**: `GIST_SECRET`
- **Value**: [paste token from Step 1]
- Click **"Add secret"**

### 4️⃣ Update Workflow Configuration

Edit `.github/workflows/ci.yml` (line 173):

```yaml
gistID: mfg-pde-coverage  # Replace with your actual gist ID from Step 2
```

Then commit and push:

```bash
git add .github/workflows/ci.yml
git commit -m "chore: Configure gist ID for coverage badge"
git push
```

## Verification

After setup:

1. ✅ Push any commit to trigger CI
2. ⏱️ Wait for CI to complete (~5-10 minutes)
3. 🔍 Check your gist - should show coverage data
4. ✨ README badge displays actual coverage percentage
5. 🎨 Badge color: Green (>80%) | Yellow (70-80%) | Red (<70%)

## Current Status

- ✅ CI workflow configured to generate coverage
- ✅ README badge URL configured
- ⏳ Awaiting: Token + Gist ID configuration

## Troubleshooting

### Badge shows "unknown"
- Check that `GIST_SECRET` is set in repository secrets
- Verify gist ID in workflow matches your actual gist
- Check CI logs for coverage badge creation step

### Badge not updating
- Ensure gist is **public** (not secret)
- Verify personal access token has `gist` scope
- Check that CI workflow completed successfully

### Coverage percentage seems wrong
- Review `codecov.yml` for ignored paths
- Check `coverage.xml` artifact in CI run
- Verify test suite is running completely

## More Information

See detailed documentation: `docs/development/CODECOV_SETUP.md`
