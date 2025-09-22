# GitHub Label System for MFG_PDE

**Date**: 2025-09-23
**Status**: ✅ **IMPLEMENTED** - Comprehensive label system established

## 🏷️ **Label Categories Overview**

The MFG_PDE repository uses a structured label system to organize issues and pull requests effectively. All labels follow consistent naming patterns and color coding.

### **📊 Priority Labels** (Required for all issues)
- `priority: high` 🔴 - High priority issues requiring immediate attention
- `priority: medium` 🟡 - Medium priority issues for planned development
- `priority: low` 🟢 - Low priority issues for future consideration

### **🎯 Area Labels** (Subject matter categorization)
- `area: algorithms` 🔵 - Mathematical algorithms and solvers
- `area: geometry` 🟦 - Mesh generation and geometric domains
- `area: performance` 🔴 - Performance optimization and benchmarking
- `area: config` ⚫ - Configuration systems and parameter management
- `area: visualization` 🟠 - Plotting and interactive visualization

### **📏 Size Labels** (Effort estimation)
- `size: small` ⚪ - Small effort: few hours to 1 day
- `size: medium` ⚪ - Medium effort: 1-3 days
- `size: large` ⚪ - Large effort: 1+ weeks

### **🔄 Status Labels** (Workflow tracking)
- `status: blocked` ⚫ - Cannot proceed due to dependencies
- `status: in-review` 🟢 - Under review or awaiting feedback

### **🔧 Type Labels** (Work categorization)
- `bug` 🔴 - Something isn't working
- `enhancement` 🔵 - New feature or request
- `documentation` 🔵 - Improvements or additions to documentation
- `type-checking` 🔵 - Type checking and mypy related improvements

### **👥 Community Labels** (Default GitHub labels)
- `good first issue` 🟣 - Good for newcomers
- `help wanted` 🟢 - Extra attention is needed
- `question` 🟣 - Further information is requested
- `duplicate` ⚪ - This issue or pull request already exists
- `invalid` 🟡 - This doesn't seem right
- `wontfix` ⚪ - This will not be worked on

## 📋 **Label Usage Guidelines**

### **🎯 Issue Labeling Requirements**
**Every issue MUST have**:
1. **One priority label**: `priority: high/medium/low`
2. **One or more area labels**: `area: algorithms/geometry/performance/config/visualization`
3. **One size label**: `size: small/medium/large`
4. **One type label**: `bug/enhancement/documentation/type-checking`

**Optional labels**:
- Status labels for workflow tracking
- Community labels for collaboration

### **✅ Good Labeling Examples**
```
Issue: "Improve convergence criteria for fixed-point iteration"
Labels: priority: medium, area: algorithms, size: small, enhancement

Issue: "Add interactive 3D mesh visualization"
Labels: priority: low, area: visualization, size: large, enhancement

Issue: "Memory leak in particle collocation solver"
Labels: priority: high, area: algorithms, area: performance, size: medium, bug
```

### **❌ Poor Labeling Examples**
```
Issue: "Fix solver"
Labels: bug
❌ Missing: priority, area, size labels
❌ Too vague: unclear what solver or what's broken

Issue: "Add new feature"
Labels: enhancement, priority: high
❌ Missing: area, size labels
❌ Too generic: no specific area identified
```

## 🔄 **Workflow Integration**

### **Issue Creation Workflow**
1. **Create issue** with descriptive title and detailed description
2. **Add required labels**: priority + area + size + type
3. **Add optional labels**: status, community as needed
4. **Assign milestone** if part of planned release
5. **Link to projects** for sprint planning

### **Pull Request Workflow**
1. **Auto-inherit labels** from linked issues when possible
2. **Add `status: in-review`** when ready for review
3. **Use area labels** to route to appropriate reviewers
4. **Remove status labels** when merged

### **Label Evolution**
- **Add new area labels** as the codebase grows (e.g., `area: ml`, `area: gpu`)
- **Retire unused labels** after major refactoring
- **Update color scheme** for consistency and accessibility

## 🎨 **Color Coding System**

**Consistent color families**:
- **Red spectrum** (`#d73a4a` to `#ff6b6b`): Urgent, bugs, performance
- **Blue spectrum** (`#0075ca` to `#1d76db`): Features, algorithms, documentation
- **Green spectrum** (`#008672` to `#c2e0c6`): Low priority, help wanted, status
- **Purple spectrum** (`#7057ff` to `#d876e3`): Community, questions
- **Orange/Yellow** (`#fbca04` to `#e67e22`): Medium priority, visualization
- **Gray/Black** (`#000000` to `#cfd3d7`): Blocked, duplicate, invalid

## 📊 **Label Analytics**

**Track repository health with label-based queries**:
```bash
# High priority open issues
gh issue list --label "priority: high" --state open

# Performance-related work
gh issue list --label "area: performance"

# Small tasks for new contributors
gh issue list --label "size: small,good first issue"

# Blocked work requiring attention
gh issue list --label "status: blocked" --state open
```

## 🔄 **Maintenance**

**Monthly label review**:
1. Check for unlabeled issues and PRs
2. Verify label consistency across similar issues
3. Update label descriptions if needed
4. Archive or consolidate redundant labels

---

**Next Steps**: Apply this labeling system to existing issues and establish automated workflows for label enforcement.