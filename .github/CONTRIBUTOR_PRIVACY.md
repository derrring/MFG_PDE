# Contributor Privacy and Personal Workflow Guide

This document explains how to maintain personal development preferences while contributing to MFG_PDE.

## 🔒 Personal Development Environment

### AI Interaction Guidelines

The repository includes a shared `CLAUDE.md` as a reference, but you can maintain personal AI guidelines:

**Personal AI Guidelines (not tracked):**
- `CLAUDE.md.personal` - Your personal Claude Code preferences
- `CLAUDE-[yourname].md` - Named personal guidelines
- `AI-GUIDELINES-personal.md` - General AI interaction preferences
- `.ai-settings/` - Directory for personal AI configuration files

**Example Personal Setup:**
```bash
# Copy the shared guidelines as a starting point
cp CLAUDE.md CLAUDE.md.personal

# Customize for your workflow
echo "# My Personal AI Preferences" >> CLAUDE.md.personal
echo "- Prefer shorter responses in development mode" >> CLAUDE.md.personal
echo "- Focus on Python typing when possible" >> CLAUDE.md.personal
```

### IDE Configuration

**Personal IDE Settings (not tracked):**
- `.vscode/settings.json.local` - Your VS Code preferences
- `.vscode/launch.json.local` - Your debug configurations
- `.idea/personal/` - IntelliJ personal settings
- `*.code-workspace.local` - Personal workspace files
- `.dev-settings/` - Any development environment settings

**Example VS Code Personal Setup:**
```json
// .vscode/settings.json.local
{
    "python.defaultInterpreterPath": "/your/python/path",
    "editor.fontSize": 14,
    "workbench.colorTheme": "Your Preferred Theme"
}
```

### Development Workflow

**Personal Development Files (not tracked):**
- `.dev-environment/` - Your personal development setup scripts
- `.personal-setup/` - Installation and configuration scripts
- `.pre-commit-config-personal.yaml` - Your pre-commit preferences
- `.dev-notes/` - Personal development notes and reminders
- `.contributor-workspace/` - Personal workspace and scratch area

## 🤝 Collaboration Guidelines

### What Stays Private
- **AI interaction preferences** - Your communication style with AI assistants
- **IDE configurations** - Editor themes, font sizes, personal shortcuts
- **Development workflow** - Your personal scripts and automation
- **Working notes** - Temporary analysis, personal reminders

### What Stays Shared
- **Core project files** - README, documentation, examples
- **CI/CD configuration** - Shared GitHub workflows and templates
- **Base configurations** - Shared VS Code settings, pre-commit config
- **Reference guidelines** - The main CLAUDE.md as a reference point

## 🚀 Getting Started with Personal Settings

1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd MFG_PDE
   ```

2. **Set up personal AI guidelines (optional)**
   ```bash
   cp CLAUDE.md CLAUDE.md.personal
   # Edit CLAUDE.md.personal with your preferences
   ```

3. **Configure personal IDE settings (optional)**
   ```bash
   cp .vscode/settings.json .vscode/settings.json.local
   # Edit .vscode/settings.json.local with your preferences
   ```

4. **Set up personal development environment (optional)**
   ```bash
   mkdir .dev-environment
   # Add your personal setup scripts
   ```

5. **Verify privacy**
   ```bash
   git status  # Should not show your personal files
   ```

## 💡 Best Practices

- **Keep the main CLAUDE.md as reference** - Don't modify it unless proposing changes for everyone
- **Use `.local` suffix** - For personal versions of shared configuration files
- **Create personal directories** - Use `.dev-*` or `.personal-*` for personal workspace areas
- **Test with git status** - Ensure your personal files aren't being tracked
- **Share improvements** - If you develop useful tools, consider contributing them back

## 🔧 Troubleshooting

**Personal files showing up in git?**
```bash
# Check if patterns are working
git check-ignore CLAUDE.md.personal  # Should output the filename
git check-ignore .vscode/settings.json.local  # Should output the filename
```

**Want to contribute personal tools back to the project?**
- Move useful scripts to appropriate project directories
- Remove personal preferences and make them generic
- Submit a PR with the contribution

This system allows everyone to work comfortably with their preferred tools and AI interaction styles while maintaining clean collaboration on the shared codebase.
