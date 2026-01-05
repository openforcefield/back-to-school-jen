# GitHub Copilot Code Review Instructions

This document provides automated review guidelines for GitHub Copilot when reviewing pull requests in this repository. The focus is on consistency, documentation standards, common technical pitfalls, and flagging items requiring human scientific review.

**Important**: Your review is automated and should highlight objective issues. Be lenient with blocking comments—the goal is to catch oversights and assist human reviewers, not to block progress on subjective matters.

---

## 1. Scope: Files to Review and Ignore

### Review These Files
- **Python scripts (*.py)**: Check for documentation, code consistency, technical issues
- **Shell scripts (*.sh)**: Verify arguments, paths, environment references
- **Text logs (*.txt)**: Scan for errors, warnings, or unexpected failures in execution logs
- **README.md files**: Ensure documentation matches code functionality

### DO NOT Review These Files
- **Binary/output files**: `*.pt`, `*.offxml`, `*.pdf`, `*.hdf5`, `*.sqlite`, `*.arrow`
- **JSON/CSV data files**: These are outputs; do not analyze content
- **Jupyter notebooks (*.ipynb)**: Discouraged in this repo; flag if present but do not review in detail
- **Large datasets**: Focus on code that generates/processes data, not the data itself

---

## 2. Documentation Consistency Checks

### Module-Level Documentation
For each **Python script**, verify:
- **Module docstring** present at the top describing overall functionality
- **Script purpose** clearly stated (what problem does it solve?)
- **Input/output behavior** documented
- If script writes custom JSON/CSV/other files, **schema must be documented** in docstring or README

### Function Documentation
For each **function** in Python scripts:
- **Function docstring** present with description of purpose
- **Parameters** documented (type hints preferred)
- **Return values** documented
- **Side effects** noted (e.g., writes files, modifies global state)

### Command-Line Interface
For scripts with CLI:
- Must use **`click`**, **`argparse`**, or similar interface (flag if hard-coded paths detected)
- Each input/output argument must have a **description** in help text
- **No hard-coded absolute paths** in code (use arguments or config files)
- Shell script (*.sh) should accompany Python script showing example usage

### Shell Scripts (*.sh)
- Document purpose at top of file (comment block)
- Include conda environment activation if applicable
- Use clear variable names
- Paths should be configurable or clearly documented

---

## 3. Code Quality and Structure

### Main Function and Workflow
- Prefer a **`main()`** function that clearly shows overall workflow
- Use **helper functions** to break down complex logic
- Main workflow should be **immediately interpretable** (well-named functions, logical flow)
- Add **comments** explaining non-obvious steps

### Assertions and Guards
- Code should include **assert statements** for critical assumptions
- Check for **guards against edge cases** with bad consequences (e.g., empty datasets, division by zero)
- Error messages are helpful but asserts are priority

### Type Hints and Clarity
- Encourage use of **type hints** for function signatures
- Variable names should be **descriptive**
- Add **logging messages** at key steps (informative, not verbose)

---

## 4. Critical Technical Checks

### 4.1 Parallelization Issues
**Check for async/parallel execution problems:**
- If using `multiprocessing`, `concurrent.futures`, `asyncio`, or similar:
  - Verify if parallelization is **mapped** (order-preserving) vs **unmapped** (unordered)
  - Flag if code assumes order but uses unordered execution (e.g., `executor.submit()` in loop without tracking futures)
  - Check if results are properly collected and matched to inputs
  - Look for potential race conditions or shared state issues

**Example red flags:**
```python
# Unmapped parallelization when order matters
for item in items:
    executor.submit(process, item)  # ⚠️ Results may be unordered
```

### 4.2 Unit Conversions
**Check for unit handling:**
- Physical quantities (energies, distances, angles, forces) should use **`openff-units`** or equivalent
- Flag any manual unit conversions (e.g., `value * 627.509` for Hartree to kcal/mol)
- Check if units are **explicitly stated** in docstrings and variable names
- Verify consistency: don't mix unit systems without conversion

**Example red flags:**
```python
energy = raw_energy * 627.509  # ⚠️ Magic number, should use openff.units
distance = coords * 0.529177  # ⚠️ Manual Bohr to Angstrom conversion
```

### 4.3 Data Filtering and Thresholds
- Check if filtering thresholds (percentiles, z-scores, etc.) are **documented and justified**
- Verify filtering doesn't accidentally remove all data (assert statements)
- Ensure filtered datasets are **logged** (how many records before/after)

### 4.4 File I/O Consistency
- Verify output files are created where documented
- Check if file paths are constructed consistently (e.g., `Path` vs string concatenation)
- Flag if overwriting files without warning or backup

### 4.5 Array Indexing and Slicing
- Check for potential **off-by-one errors**
- Verify indices are within bounds (especially if using hard-coded indices)
- Flag if assuming array shapes without validation

### 4.6 Unmentioned Critical Technical Issues
- Check the code for any technical issues not mentioned here that are overlooked or inconsistent with the purpose of the code

---

## 5. Repository-Specific Checks

### Workflow Directory Structure
This repo follows a numbered workflow structure (`1_data/`, `2_filtered_results/`, etc.):
- Check if new code fits the expected directory pattern
- Verify input/output paths reference correct directories
- Ensure logs are written to appropriate locations

### Environment Files
- If modifying dependencies, check if `environment.yaml` or `environment_full.yaml` updated
- Flag if importing packages not listed in environment files

### README Consistency
- If adding new scripts or workflows, verify `README.md` updated
- Check if README accurately describes script functionality
- The guidelines of this repo are specified in `./README.md`

---

## 6. Flag for Human Scientific Review

After completing automated checks, create a **"Requires Scientific Review"** section highlighting:

### Scientific Assumptions and Methods
- **Parametrization decisions**: Changes to force field parameters, optimization methods, basis sets
- **Data filtering criteria**: Thresholds, outlier removal, train/test splits
- **Model architecture**: Neural network layers, activation functions, training hyperparameters
- **Physical chemistry**: Energy calculations, conformer generation, SMILES handling

### Specific Items to Flag
1. **File and line numbers** where scientific decisions are made
2. **Changes to algorithms** or computational methods
3. **New thresholds or hyperparameters** introduced
4. **Statistical assumptions** (e.g., normality, independence)
5. **Validation metrics** and acceptance criteria

### Flag Suggested Scientific Issues
- When flagging specific scientific items, include a line on whether the change is expected to be an issue scientifically, leaving judgement to another reviewer

### Example Format
```
## 🔬 Requires Scientific Review

1. **`filter_data.py:45-67`**: New z-score threshold of 3.0 for energy outlier removal. Verify this is appropriate for SPICE-2 dataset.

2. **`fit_data.py:123-145`**: Changed optimizer from Adam to SGD. Review impact on convergence.

3. **`make_offxml.py:89`**: Bond parameter assignment logic modified. Verify chemical correctness.
```

---

## 7. Review Output Format

Structure your review as follows:

### Summary
- Brief overview (2-3 sentences)
- Overall assessment: ✅ Minor issues, ⚠️ Several issues, 🚨 Critical issues

### Documentation Issues
- List missing docstrings, unclear descriptions, undocumented schemas

### Technical Issues
- List parallelization problems, unit conversion issues, potential bugs
- Include file paths and line numbers

### Code Quality Suggestions
- Non-blocking improvements (type hints, better variable names, etc.)

### 🔬 Requires Scientific Review
- Flagged items for human expert review with file/line references

---

## 8. Example Review Comment

```markdown
## GitHub Copilot Automated Review

### Summary
The PR adds a new filtering script for SPICE data. Documentation is mostly complete, but found potential unit conversion issue and one section requiring scientific review.

### Documentation Issues
- `filter_data.py`: Missing module-level docstring describing filtering strategy
- `run_filter_data.sh`: No comments explaining threshold choice

### Technical Issues
⚠️ **`filter_data.py:78`**: Manual unit conversion `energy * 627.509` detected. Use `openff.units` instead:
```python
from openff.units import unit
energy = raw_energy * unit.hartree
energy_kcal = energy.to(unit.kilocalorie_per_mole)
```

✅ Parallelization looks correct (using `executor.map()` preserves order)

### Code Quality Suggestions
- Consider adding type hints to `process_molecule()` function
- Add assertion after filtering to ensure dataset not empty

### 🔬 Requires Scientific Review
1. **`filter_data.py:45-67`**: Z-score threshold set to 2.5 for energy outlier removal. Previous work used 3.0. Verify this change is intentional and appropriate.
2. **`filter_data.py:112`**: Force calculation now includes non-bonded terms. Review if this aligns with training objective.
```

---

## Notes
- This automated review focuses on **consistency and common pitfalls**, not scientific correctness
- Human reviewers should validate all scientific assumptions and methods
- When in doubt, flag for human review rather than blocking
