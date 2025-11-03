# Claude Code Guidelines

## Critical: Git Commit Protocol

**ALWAYS verify that changes are committed and pushed to git before considering work complete.**

### Failure Mode Identified: 2025-11-03

On 2025-11-03, significant work (stochastic photon sampling functions in SpectralFunctions.py) was lost because:
1. Functions were implemented and tested successfully
2. Work was never committed to git (only existed in working directory)
3. A `git checkout` command to fix a broken edit reverted ALL uncommitted changes
4. Hours of work had to be re-implemented from documentation

### Required Workflow

1. **After implementing each significant feature:**
   - Check git status: `git status`
   - Add changed files: `git add <files>`
   - Commit with descriptive message: `git commit -m "..."`
   - Verify commit: `git log -1`

2. **Before reverting ANY file:**
   - Check what will be lost: `git diff <file>`
   - Consider `git stash` instead of `git checkout` to preserve work
   - Only use `git checkout` if you're certain you want to discard changes

3. **At end of session or before risky operations:**
   - Commit all working code
   - Push to remote if appropriate: `git push`

### Never Assume

- Never assume code is "saved" just because it's in a file
- Never assume code is committed just because it works
- Never assume `git checkout` will only revert "recent" changes
- ALWAYS explicitly verify git status before and after operations

## General Guidelines

- Test thoroughly before committing
- Write clear commit messages
- Document significant architectural changes
- Keep documentation (*.md files) in sync with code changes
