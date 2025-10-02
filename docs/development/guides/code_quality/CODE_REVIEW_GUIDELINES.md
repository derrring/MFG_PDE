# Code Review Guidelines for MFG_PDE

**Date**: 2025-09-23
**Status**: ✅ **ACTIVE** - Standard code review practices

## 🎯 **Code Review Philosophy**

**Goal**: Maintain high code quality while supporting research development flexibility.

**Principles**:
- **Constructive feedback** over criticism
- **Educational approach** - explain the "why" behind suggestions
- **Research-friendly** - balance rigor with development speed
- **Collaborative** - reviewer and author work together toward quality

## 📋 **Review Process Workflow**

### **1. Pre-Review (Author)**
```bash
# Before creating PR
git checkout -b feature/descriptive-name
# Make changes, test locally
ruff check changed_files.py  # Style check
mypy changed_files.py --ignore-missing-imports  # Type check
pytest tests/  # Run tests

# Create PR with proper labels
gh pr create --title "Clear descriptive title" \
  --label "priority: medium,area: algorithms,size: small,enhancement"
```

### **2. Review Request**
```bash
# Author requests review
gh pr create --reviewer derrring --draft  # If early feedback needed
# OR
gh pr create --reviewer derrring         # If ready for final review
```

### **3. Review Execution (Reviewer)**
```bash
# Checkout PR locally for testing
gh pr checkout [PR_NUMBER]
# Test the changes
python -m pytest tests/
# Review files
gh pr view [PR_NUMBER] --json files --jq '.files[].filename'
```

### **4. Review Feedback**
```bash
# Provide feedback
gh pr review [PR_NUMBER] --comment --body "Detailed feedback here"
gh pr review [PR_NUMBER] --request-changes --body "Please address X before merging"
gh pr review [PR_NUMBER] --approve --body "LGTM! Ready to merge"
```

## 🔍 **What to Review**

### **✅ Must Review**
1. **Functionality**: Does code accomplish stated goals?
2. **Correctness**: Mathematical accuracy for algorithms
3. **Style**: Follows project conventions (Ruff compliance)
4. **Type Safety**: Proper type annotations where applicable
5. **Tests**: Adequate coverage for new functionality
6. **Documentation**: Clear docstrings and comments
7. **Breaking Changes**: Properly documented and justified

### **⚠️ Important to Check**
- **Performance**: No obvious bottlenecks in hot paths
- **Memory Usage**: Reasonable memory consumption for large problems
- **Backward Compatibility**: Research code continues to work
- **API Design**: Intuitive interfaces for users
- **Error Handling**: Graceful handling of edge cases

### **ℹ️ Nice to Have**
- **Code Organization**: Logical structure and modularity
- **Naming**: Clear, descriptive variable and function names
- **Comments**: Explain complex mathematical concepts

## 📊 **Review Criteria by PR Type**

### **🔧 Bug Fixes**
- **Root Cause**: Is the underlying issue addressed?
- **Test Coverage**: Does test prevent regression?
- **Side Effects**: Any unintended consequences?

### **✨ New Features**
- **Design**: Is the approach sound?
- **Integration**: How does it fit with existing code?
- **Documentation**: Are examples provided?
- **Performance**: Scalability considerations?

### **🏗️ Refactoring**
- **Scope**: Is refactoring focused and logical?
- **Equivalence**: Does behavior remain the same?
- **Benefits**: Clear improvement in maintainability?

### **📚 Documentation**
- **Accuracy**: Does documentation match code?
- **Completeness**: Are examples clear and working?
- **Accessibility**: Appropriate for target audience?

## 🚨 **Review Standards**

### **❌ Must Reject If**
- Tests are failing
- Introduces security vulnerabilities
- Breaks backward compatibility without justification
- Lacks type annotations for new public APIs
- Missing tests for new functionality
- Code style violations (Ruff failures)

### **⚠️ Request Changes If**
- Performance issues in critical paths
- Unclear or missing documentation
- Complex code without explanatory comments
- Non-intuitive API design
- Missing error handling for edge cases

### **✅ Approve If**
- All CI checks pass
- Code accomplishes stated goals
- Follows project conventions
- Has adequate test coverage
- Documentation is clear and complete
- No breaking changes or properly documented

## 💬 **Feedback Guidelines**

### **✅ Good Feedback**
```markdown
**Specific**: "Line 42: Consider using `numpy.trapezoid` instead of `numpy.trapz` for NumPy 2.0+ compatibility"

**Educational**: "The current approach has O(n²) complexity. Consider using a hash map to reduce to O(n)"

**Constructive**: "This works correctly. For better readability, consider extracting the validation logic into a separate function"
```

### **❌ Poor Feedback**
```markdown
**Vague**: "This doesn't look right"
**Critical**: "This code is terrible"
**Nitpicky**: "Use single quotes instead of double quotes" (when both are acceptable)
```

## 🎯 **Special Considerations for MFG_PDE**

### **Research Code Flexibility**
- **Accept** `**kwargs: Any` patterns for research interfaces
- **Allow** experimental code in examples/ with clear documentation
- **Prioritize** mathematical correctness over perfect style

### **Mathematical Accuracy**
- **Verify** algorithm implementations match cited papers
- **Check** numerical stability considerations
- **Validate** boundary condition handling

### **Performance Awareness**
- **Consider** impact on large-scale problems
- **Check** memory allocation patterns
- **Verify** GPU compatibility where applicable

## 📋 **Review Checklist Template**

```markdown
## Code Review Checklist

### Functionality
- [ ] Code accomplishes stated goals
- [ ] Mathematical algorithms are correct
- [ ] Edge cases are handled appropriately

### Quality
- [ ] Follows project style guidelines
- [ ] Has appropriate type annotations
- [ ] Includes comprehensive tests
- [ ] Documentation is clear and complete

### Integration
- [ ] No breaking changes (or properly documented)
- [ ] Backward compatibility maintained
- [ ] Fits well with existing architecture

### Performance
- [ ] No obvious performance issues
- [ ] Memory usage is reasonable
- [ ] Scalable for large problems

### Review Decision
- [ ] **Approve**: Ready to merge
- [ ] **Request Changes**: Issues must be addressed
- [ ] **Comment**: Suggestions for improvement
```

## 🔄 **Post-Review Process**

### **After Approval**
```bash
# Merge with appropriate method
gh pr merge [PR_NUMBER] --squash  # For feature branches
gh pr merge [PR_NUMBER] --merge   # For maintaining commit history
```

### **After Merge**
- Update related issues
- Create follow-up issues if needed
- Document any new patterns or decisions

---

**Remember**: Code review is a **collaborative process** aimed at maintaining quality while supporting research innovation. Focus on being helpful, educational, and constructive!
