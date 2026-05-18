# Claude Code Instructions for pyS3M

**Repository:** pyS3M - Python package for multicolour single-molecule localization microscopy
**Last Updated:** November 3, 2025

---

## 🚨 CRITICAL: File Locations

**ALWAYS look in the `claude/` directory for:**
- `TODO.md` - Active task list
- `LOG.md` - Completed work archive
- `CLAUDE.md` - This file (instructions for Claude)

**Do NOT create these files in the project root!**

---

## 🚨 CRITICAL: Git Commit Protocol

**ALWAYS verify that changes are committed to git before considering work complete.**

### Failure Mode Identified: 2025-11-03

Significant work (stochastic photon sampling, ~300 lines) was lost because:
1. ✓ Functions implemented and tested successfully (867× speedup confirmed)
2. ✗ Work was NEVER committed to git (only in working directory)
3. ✗ `git checkout` command reverted ALL uncommitted changes
4. ⚠️ Hours of work re-implemented from documentation

**Lesson: Working code in files ≠ Saved code in git**

### Required Workflow

#### After implementing each significant feature:
```bash
git status              # Check what changed
git diff <file>         # Review changes
git add <files>         # Stage changes
git commit -m "..."     # Commit with message
git log -1 --stat       # VERIFY commit succeeded
```

#### Before reverting ANY file:
```bash
git diff <file>                    # Check what will be lost
git stash push -m "..." <file>     # Consider stashing
git checkout <file>                # ⚠️ PERMANENT DATA LOSS
```

#### Never Assume:
- ❌ Code is saved just because it's in a file
- ❌ Code is committed just because tests pass
- ❌ `git checkout` only reverts "recent" changes
- ✅ VERIFY with `git status` and `git log -1`

---

## General Workflow Instructions

### 1. Documentation Management

#### TODO.md - Active Tasks Only
- Keep TODO.md **minimal** and focused on **pending tasks only**
- Remove completed items immediately after finishing
- Structure: High-level task categories with brief descriptions
- Include estimated complexity/time only for pending work

#### LOG.md - Completed Work Archive
- **Move completed tasks from TODO.md to LOG.md**
- Add detailed session notes for each work session
- Include:
  - Date and session title
  - Detailed description of changes
  - Files modified with line counts
  - Performance metrics/benchmarks if applicable
  - Test results
  - Next steps/follow-ups
- Keep most recent session at the top

#### When Completing Tasks:
1. ✅ Mark task as complete in TODO.md
2. 📝 Document details in LOG.md with full context
3. ✂️ Remove completed section from TODO.md
4. 🔄 Keep TODO.md focused on what's next

### 2. Code Organization

#### Module Structure
- `src/` - Main source code
- `unit_tests/` - Test files
- `unit_tests/claude/` - Claude-specific test utilities
- `claude/` - Development documentation (TODO.md, LOG.md, CLAUDE.md)

#### Import Management
- All imports should use `ImportManager.py` for consistency
- Use lazy loading for local modules to avoid circular imports
- External dependencies loaded through `get_module()` helper

#### Plotting Standards
- Use `PlottingBase.py` classes for all plotting:
  - `PublicationPlotter` - Publication-quality figures
  - `AnalysisPlotter` - Interactive analysis with large dataset support
  - `DriftPlotter` - Drift correction specific plots
- **CRITICAL:** When implementing plots, ALWAYS check PlottingBase.py first to:
  - Verify which methods actually exist (don't assume plotter.colors, plotter.save_figure, etc.)
  - Check the correct API for create_subplots, create_figure, etc.
  - See how existing code (DriftPlotting.py) uses PlottingBase
  - Use standard matplotlib (plt.subplots) if PlottingBase doesn't have the method
- Leverage `DatashaderMixin` for datasets >10k points
- Always use `rasterized=True` for large scatter plots

### 3. Testing Approach

#### Test Organization
- Core tests in `unit_tests/`
- Development/experimental tests in `unit_tests/claude/`
- Always create benchmarks for performance-critical code
- Use synthetic data generators for reproducible tests

#### Before Committing
- Run relevant tests to ensure no breakage
- Check import warnings with `python -c "import sys; sys.path.insert(0, 'src'); import DriftCorrectionFunctions"`
- Verify no circular import errors

### 4. Performance Optimization Guidelines

#### When Optimizing Plotting:
- Threshold: Consider datashader for >10k points
- Downsampling: Use density-aware sampling for better representation
- Rasterization: Always enable for matplotlib scatter with >1k points
- Multi-dataset: Use `plot_multi_dataset_scatter()` for grouped data

#### When Optimizing Processing:
- Vectorize operations using NumPy
- Use numba JIT for tight loops
- Consider memory mapping for large datasets
- Profile before optimizing (use `time` benchmarks)

### 5. Code Quality Standards

#### Docstrings
- Use Google-style docstrings
- Include Args, Returns, Examples sections
- Add performance notes for complex methods
- Document expected data formats (especially for recarrays)

#### Type Hints
- Use type hints for all public methods
- Import from `typing` module
- Use `Optional` for nullable parameters
- Document units in parameter descriptions (nm, pixels, etc.)

#### Error Handling
- Provide helpful error messages with context
- Use try/except with specific exceptions
- Always clean up resources (close figures, etc.)
- Add traceback printing for debugging complex failures

### 6. Git Workflow

#### Commit Messages
- Use conventional commits format: `type(scope): description`
- Types: `feat`, `fix`, `refactor`, `perf`, `test`, `docs`
- Examples:
  - `feat(plotting): add datashader integration for large datasets`
  - `perf(plotting): optimize multi-dataset rendering with 10x speedup`
  - `refactor(drift): extract plotting methods to DriftPlotting.py`

#### Branch Strategy
- Work on `main` branch for this project
- Large refactors: mention in commit message
- Keep commits atomic and focused

### 7. Repository-Specific Notes

#### Virtual Environment
- Use `~/.virtualenvs/pyS3M/bin/python` for all commands
- Dependencies include: numpy, scipy, matplotlib, numba
- Optional: datashader, pandas (for large dataset optimization)

#### Key Files to Understand
- `DriftCorrectionFunctions.py` - Main drift correction algorithms
- `PlottingBase.py` - Base plotting classes and mixins
- `ImportManager.py` - Centralized import management
- `FiducialDetection.py` - Automatic fiducial detection pipeline
- `SM_extractionfunctions.py` - Single molecule extraction and mixture analysis (GMM, robust covariance fitting)

#### Performance Expectations
- Drift correction: <1s per 1000 localisations
- Plotting: <20ms for <100k points
- Rendering: Use datashader for >10k points
- Memory: <2GB for typical datasets (~100k localisations)

### 8. Session Workflow Template

```markdown
## Session: [Date] - [Title]

### [Task] ✅ COMPLETE

**Summary:** Brief overview of what was accomplished

**Key Achievements:**
1. **[Component]** (filename)
   - Added X method (N lines)
   - Optimized Y functionality
   - Fixed Z issue

2. **[Component]** (filename)
   - Implementation details
   - Performance improvements

**Files Modified:**
- `path/to/file.py` (+X lines, description)

**Performance Metrics:**
[Benchmark results table or bullet points]

**Next Steps:**
- Follow-up task 1
- Follow-up task 2
```

### 9. Common Patterns

#### Creating New Plotters
```python
from PlottingBase import AnalysisPlotter

plotter = AnalysisPlotter(datashader_threshold=10000)
fig, ax = plotter.create_figure()
plotter.plot_large_scatter(ax, x, y, threshold=10000)
plotter.save_or_show(fig, save_path="output.png")
```

#### Multi-Dataset Plotting
```python
datasets = [
    {'x': data1['xc'], 'y': data1['yc']},
    {'x': data2['xc'], 'y': data2['yc']}
]
plotter.plot_multi_dataset_scatter(
    ax, datasets,
    labels=['Group 1', 'Group 2'],
    threshold=10000,
    rasterized=True
)
```

#### Performance Benchmarking
```python
import time
times = []
for trial in range(3):
    start = time.time()
    # ... operation to benchmark ...
    times.append(time.time() - start)
print(f"Average: {np.mean(times)*1000:.1f}ms ± {np.std(times)*1000:.1f}ms")
```

---

## Quick Reference

### File Locations
- Documentation: `claude/TODO.md`, `claude/LOG.md`
- Source code: `src/*.py`
- Tests: `unit_tests/` and `unit_tests/claude/`
- Virtual env: `~/.virtualenvs/pyS3M/`

### Important Commands
```bash
# Run tests
~/.virtualenvs/pyS3M/bin/python unit_tests/claude/test_plotting_performance.py

# Check imports
~/.virtualenvs/pyS3M/bin/python -c "import sys; sys.path.insert(0, 'src'); import DriftCorrectionFunctions"

# Run specific test
~/.virtualenvs/pyS3M/bin/python unit_tests/claude/test_drift_correction.py
```

### Contact & Context
- Project owner: J. Beckwith
- Primary focus: SMLM drift correction and analysis
- Code style: PEP 8 with focus on readability and performance

---

**Remember:** Keep TODO.md minimal by moving completed work to LOG.md immediately after finishing tasks!
