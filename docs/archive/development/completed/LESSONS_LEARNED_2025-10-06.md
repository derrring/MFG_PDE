# Lessons Learned - Software Engineering Session

**Date**: October 6, 2025
**Session**: Type Safety Improvements
**Issue**: Branch Structure Violation

---

## ⚠️ What Went Wrong

**Violation**: Committed directly to `main` branch instead of using hierarchical branch structure

**What happened**:
- Made 9 commits directly to `main`
- No parent/child branch structure
- Skipped proper merge workflow

**According to CLAUDE.md** (lines 443-543):
> **Always Work on Branches**
> - ❌ Never commit directly to `main`
> - ✅ Create feature branches for all work
> - ✅ Use parent branches to organize related work

---

## ✅ What Should Have Been Done

### Correct Workflow for Type Safety Work:

```bash
# Step 1: Create parent branch
git checkout main
git checkout -b chore/type-safety-improvements

# Step 2: Work in child branches
git checkout -b chore/infrastructure-tooling
# commits: Makefile, Dependabot, CLI
git checkout chore/type-safety-improvements
git merge chore/infrastructure-tooling --no-ff
git push

git checkout -b chore/remove-unused-ignores
# commits: remove 80 type ignores
git checkout chore/type-safety-improvements
git merge chore/remove-unused-ignores --no-ff
git push

git checkout -b chore/add-type-annotations
# commits: maze_config, factory, variables
git checkout chore/type-safety-improvements
git merge chore/add-type-annotations --no-ff
git push

# Step 3: Merge to main when complete
git checkout main
git merge chore/type-safety-improvements --no-ff
git push
```

---

## 📚 Why It Matters

**Benefits of hierarchical branch structure**:
1. **Organized History**: Related changes grouped logically
2. **Easy Rollback**: Can revert entire feature set by reverting parent merge
3. **Clear Progress**: Parent branch shows cumulative progress
4. **Clean Main**: Main only receives complete, tested feature sets

**For software engineering work specifically**:
- Multiple related improvements naturally fit child branches
- Each phase (infrastructure, cleanup, annotations) could be a child
- Parent branch provides integration testing point
- Main stays stable

---

## 🎯 Action Items for Future Work

### Mandatory Process (CLAUDE.md compliance):

**Before starting any multi-step work**:
1. ✅ Create parent branch: `<type>/descriptive-name`
   - Types: `feature/`, `fix/`, `chore/`, `docs/`, `refactor/`, `test/`
2. ✅ Create child branches for logical subtasks
3. ✅ Merge child → parent (test integration)
4. ✅ Only merge parent → main when all work complete

**Examples for common scenarios**:
```bash
# Type safety work
chore/type-safety-improvements (parent)
  ├── chore/remove-unused-ignores (child)
  ├── chore/add-function-annotations (child)
  └── chore/add-variable-annotations (child)

# Feature development
feature/semi-lagrangian-solver (parent)
  ├── feature/sl-core-implementation (child)
  ├── feature/sl-tests (child)
  └── feature/sl-documentation (child)

# Bug fixes
fix/convergence-issues (parent)
  ├── fix/hjb-convergence (child)
  └── fix/fp-convergence (child)
```

---

## 🔄 Current State Resolution

**Decision**: Accept current state
- Work is already pushed to `main`
- Solo development context
- Educational value in documenting the mistake
- No need to rewrite history

**Commitment**: All future work will follow hierarchical branch structure

---

## 📋 Checklist for Next Session

Before starting work:
- [ ] Is this multi-step work? (If yes → use branches)
- [ ] Created parent branch from main?
- [ ] Working in child branch (not parent)?
- [ ] Tested in parent branch before merging to main?
- [ ] Branch names follow `<type>/descriptive-name` pattern?

**Never**:
- [ ] ❌ Commit directly to main
- [ ] ❌ Create branches without clear parent
- [ ] ❌ Use generic branch names (dev, temp, work)

---

## 💡 Key Takeaway

> **For software engineering projects, we should follow CLAUDE.md principles even more strictly than for research code.**
>
> Branch structure provides organization and safety that becomes critical when making systematic changes like type safety improvements.

---

**Lesson Status**: ✅ Documented
**Future Compliance**: Mandatory
**Reference**: CLAUDE.md lines 443-568 (Hierarchical Branch Structure)
