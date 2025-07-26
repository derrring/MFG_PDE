# MFG_PDE Documentation and Code Formatting Preferences

**Date**: July 26, 2025  
**Purpose**: Document formatting and style preferences for consistent development

## Text Formatting Preferences

### Emoji and Unicode Usage
- **NO EMOJI**: Avoid using emoji symbols (🎯, ✅, 📊, etc.) in documentation, code, or comments
- **UTF-8 Alternatives**: Use appropriate UTF-8 text alternatives instead:
  - Use `[✓]` instead of ✅
  - Use `[PLAN]` instead of 📋
  - Use `[HIGH]`, `[MEDIUM]`, `[LOW]` instead of ⭐⭐⭐
  - Use `[YES]`, `[NO]`, `[OPTIONAL]` instead of ✅/❌/⚪
  - Use `[DONE]`, `[COMPLETED]`, `[IN PROGRESS]` for status indicators

### Mathematical Content and LaTeX Support
- **FULL LaTeX SUPPORT**: Freely use LaTeX mathematical notation in all contexts
- **Documentation**: Use LaTeX math expressions in markdown: `$\Delta t$`, `$$\frac{\partial U}{\partial t}$$`
- **Plot Labels**: Use LaTeX for mathematical expressions in plot legends and annotations
- **Comments**: Include LaTeX mathematical notation in code comments when explaining algorithms
- **Pseudocode**: Use LaTeX mathematical symbols in algorithm descriptions
- **Professional Mathematical Communication**: Prioritize precise mathematical notation over simplified text

### Section Headers
- Use clean text-based section headers
- Avoid decorative symbols in headers
- Use meaningful descriptive text

### Status Indicators
- Use bracketed text format: `[STATUS]`
- Common status indicators:
  - `[✓]` - Completed
  - `[PLAN]` - Planning phase
  - `[DONE]` - Finished
  - `[HIGH]` - High priority
  - `[MEDIUM]` - Medium priority
  - `[LOW]` - Low priority
  - `[YES]` - Recommended
  - `[NO]` - Not recommended
  - `[OPTIONAL]` - Optional enhancement
  - `[CONSIDER]` - Worth considering
  - `[RESEARCH]` - Research phase
  - `[FUTURE]` - Future enhancement

## Library and Tool Preferences

### Terminal Output
- **NO Rich Library**: Do not use the Rich library for terminal output
- **Standard Logging**: Use Python's standard logging module
- **Optional Colorlog**: colorlog is acceptable for colored terminal output
- **Simple Progress**: tqdm is acceptable for progress bars

### Rationale
These preferences ensure:
- Consistent formatting across all documentation
- Better compatibility across different terminal environments
- Cleaner, more professional appearance
- Easier maintenance and updates
- Universal readability regardless of emoji support

## Application Guidelines

1. **New Documentation**: Follow these preferences for all new documents
2. **Code Comments**: Use text-based status indicators in code comments
3. **Git Commits**: Use text-based indicators in commit messages
4. **Progress Messages**: Use simple text for user-facing messages
5. **Error Messages**: Use clear text without emoji decorations

## Examples

### Good Examples:
```markdown
## Implementation Status: [✓] COMPLETED
### High Priority Items [HIGH]
- Task completed [DONE]
- Feature ready for testing [READY]
```

### Avoid:
```markdown
## 🎯 Implementation Status: ✅ COMPLETED
### 🚀 High Priority Items ⭐⭐⭐
- Task completed ✅
- Feature ready for testing 🧪
```

---

**Note**: This preference document should be referenced when making documentation updates or code modifications to ensure consistent formatting across the MFG_PDE project.