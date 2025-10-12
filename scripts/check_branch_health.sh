#!/bin/bash
# Branch Health Check Script
# Provides dynamic assessment of repository branch state
#
# Usage: bash scripts/check_branch_health.sh

set -e

echo "============================================"
echo "   MFG_PDE Branch Health Report"
echo "============================================"
echo ""

# Fetch latest refs
git fetch --prune origin >/dev/null 2>&1

# Count branches by type
echo "=== Branch Count by Type ==="
total_branches=0
for type in feature fix chore docs refactor test research; do
  count=$(git branch -r | grep "origin/$type/" | wc -l | tr -d ' ')
  if [ "$count" -gt 0 ]; then
    echo "  $type: $count branches"
    total_branches=$((total_branches + count))
  fi
done
echo "  TOTAL: $total_branches branches (excluding main)"
echo ""

# Count open PRs
open_prs=$(gh pr list --state open --json number --jq '. | length')
echo "=== Open Pull Requests ==="
echo "  Total: $open_prs PRs"
echo ""

# Find stale branches (no activity in 7+ days)
echo "=== Stale Branches (>7 days no activity) ==="
stale_prs=$(gh pr list --state open --json number,title,headRefName,updatedAt --jq \
  'map(select((now - (.updatedAt | fromdateiso8601)) > (7*24*3600)))')

stale_count=$(echo "$stale_prs" | jq '. | length')
if [ "$stale_count" -gt 0 ]; then
  echo "$stale_prs" | jq -r '.[] | "  PR #\(.number): \(.headRefName) - \(.title)"'
  echo "  Total: $stale_count stale PRs"
else
  echo "  ✅ No stale PRs"
fi
echo ""

# Find branches significantly behind main
echo "=== Branches Behind Main (>10 commits) ==="
behind_count=0
for branch in $(git branch -r | grep -v HEAD | grep -v main); do
  behind=$(git rev-list --count "$branch..origin/main" 2>/dev/null || echo "0")
  if [ "$behind" -gt 10 ]; then
    echo "  $branch: $behind commits behind"
    behind_count=$((behind_count + 1))
  fi
done
if [ "$behind_count" -eq 0 ]; then
  echo "  ✅ All branches reasonably up-to-date"
fi
echo ""

# Merged branches that still exist
echo "=== Merged Branches Needing Cleanup ==="
merged_branches=$(git branch -r --merged origin/main | grep -v "HEAD\|main" || true)
merged_count=$(echo "$merged_branches" | grep -c "origin/" | tr -d '\n' || echo "0")
merged_count=${merged_count:-0}  # Ensure it's a valid number
if [ "$merged_count" -gt 0 ] 2>/dev/null; then
  echo "$merged_branches" | head -10
  if [ "$merged_count" -gt 10 ]; then
    echo "  ... and $((merged_count - 10)) more"
  fi
  echo "  Total: $merged_count merged branches"
  echo ""
  echo "  Cleanup command:"
  echo "  gh pr list --state merged --limit $merged_count --json headRefName \\"
  echo "    --jq '.[].headRefName' | xargs -I {} git push origin --delete {}"
else
  echo "  ✅ No merged branches to clean up"
fi
echo ""

# Health assessment and recommendations
echo "============================================"
echo "   Health Assessment"
echo "============================================"
echo ""

# Determine project phase based on branch patterns
if [ "$total_branches" -le 5 ]; then
  phase="Maintenance"
  status="✅ HEALTHY"
elif [ "$total_branches" -le 8 ]; then
  phase="Active Development"
  status="✅ HEALTHY"
elif [ "$total_branches" -le 12 ]; then
  phase="Major Work / Refactor"
  status="⚡ MODERATE"
else
  phase="Unknown (High Complexity)"
  status="⚠️  HIGH"
fi

echo "Project Phase: $phase"
echo "Branch Count: $total_branches"
echo "Status: $status"
echo ""

echo "=== Recommendations ==="

if [ "$stale_count" -gt 2 ]; then
  echo "🚨 PRIORITY: Address $stale_count stale PRs"
  echo "   - Review and merge ready PRs"
  echo "   - Close abandoned work"
  echo "   - Rebase and update active work"
  echo ""
fi

if [ "$merged_count" -gt 5 ] 2>/dev/null; then
  echo "⚠️  Clean up $merged_count merged branches"
  echo "   Run: gh pr list --state merged --json headRefName --jq '.[].headRefName' | xargs -I {} git push origin --delete {}"
  echo ""
fi

if [ "$behind_count" -gt 2 ]; then
  echo "⚠️  $behind_count branches significantly behind main"
  echo "   - Rebase long-lived branches"
  echo "   - Consider merging or closing outdated work"
  echo ""
fi

if [ "$total_branches" -gt 12 ]; then
  echo "⚠️  Branch count HIGH ($total_branches)"
  echo "   - Prioritize merging completed work"
  echo "   - Consolidate related branches"
  echo "   - Avoid creating new branches until count < 10"
  echo ""
elif [ "$total_branches" -gt 8 ]; then
  echo "⚡ Branch count MODERATE ($total_branches)"
  echo "   - Review stale branches weekly"
  echo "   - Consider hierarchical branch structure for complex work"
  echo ""
else
  echo "✅ Branch management is healthy"
  echo "   - Current state supports productive development"
  echo "   - Continue monitoring weekly"
  echo ""
fi

# Velocity metrics (if gh cli supports it)
echo "=== Recent Activity ==="
recent_merges=$(gh pr list --state merged --limit 10 --json mergedAt --jq '. | length')
echo "  Last 10 merged PRs: Check with 'gh pr list --state merged --limit 10'"
echo "  Recent merge rate: ~$((recent_merges)) PRs"
echo ""

echo "============================================"
echo "   Run this script weekly for best results"
echo "============================================"
