---
title: "Rewrite DOCX table generator from Node.js to Python for architecture consistency"
date: 2026-03-18
category: integration-issues
tags:
  - python-docx
  - architecture-mismatch
  - polars
  - jama-formatting
  - xml-manipulation
  - code-review
components:
  - scripts/generate_docx_tables.py
  - Makefile
  - .gitignore
  - CLAUDE.md
severity: low
time_to_resolve: single-session
root_cause_type: architecture-mismatch
---

# Rewrite DOCX Table Generator from Node.js to Python

## Problem

A Node.js script (`scripts/generate_docx_tables.js`) using the `docx` npm package was generating JAMA-formatted Word tables from CSV outputs. This was an architecture mismatch: the entire project is Python-only (uv, Polars, marimo, python-docx already declared in `pyproject.toml` but unused). Building DOCX tables required a separate Node.js runtime and npm, adding toolchain complexity.

## Root Cause

The original implementation chose Node.js because the `docx` npm library has a clean declarative API. However, Python's `python-docx` (already a declared dependency) provides equivalent functionality -- it just requires XML manipulation for fine-grained control over borders, margins, and spacing that the high-level API does not expose.

## Solution

Rewrote the generator as `scripts/generate_docx_tables.py` using `python-docx` and `polars`. Deleted the JS script. Updated the Makefile `docx` target to use `uv run`.

Code review (5 parallel agents: Python reviewer, security sentinel, architecture strategist, code simplicity, pattern recognition) found 12 additional issues, all fixed:

| # | Finding | Fix |
|---|---|---|
| 1 | Pyright: `doc: Document` type errors | Import `DocumentType` from `docx.document` for signatures |
| 2 | Pyright: `BaseStyle.font` not exposed | `# type: ignore[union-attr]` with explanation |
| 3 | Unused returns from `build_table`/`add_table_title` | Removed returns, annotated `-> None` |
| 4 | `.gitignore` had `output/` not `outputs/` | Added `outputs/` |
| 5 | Makefile `clean` used `output/` | Added `outputs/` to rm |
| 6 | `NONE` shadowed Python builtin | Renamed to `NO_BORDER` |
| 7 | `THIN` unclear name | Renamed to `THIN_BORDER` |
| 8 | `set_cell_borders` missing remove-existing | Added `findall`/`remove` before `append` |
| 9 | Wrong enum for table alignment | `WD_ALIGN_PARAGRAPH` -> `WD_TABLE_ALIGNMENT` |
| 10 | Missing type annotations on XML helpers | Added `cell: Any`, `-> None` |
| 11 | Trivial wrapper functions | Inlined `build_manuscript`/`build_supplement` into `main()` |
| 12 | `scripts/` not in pyright scope or CLAUDE.md | Added to both |

## Key Code Patterns

### 1. XML Helper Pattern for python-docx

`python-docx` does not expose cell borders, cell margins, or exact line spacing at the high-level API. You must construct Open XML elements directly. The critical pattern is: get the parent properties element, **remove any existing child of the same type**, then append the new element.

```python
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

THIN_BORDER = {"sz": "4", "val": "single", "color": "000000"}
NO_BORDER = {"sz": "0", "val": "none", "color": "auto"}

def set_cell_borders(cell: Any, **kwargs: dict[str, str]) -> None:
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    borders = OxmlElement("w:tcBorders")
    for edge, attrs in kwargs.items():
        el = OxmlElement(f"w:{edge}")
        for k, v in attrs.items():
            el.set(qn(f"w:{k}"), str(v))
        borders.append(el)
    # Remove existing borders before appending (prevents duplicate XML elements)
    for existing in tc_pr.findall(qn("w:tcBorders")):
        tc_pr.remove(existing)
    tc_pr.append(borders)
```

The same remove-then-append pattern applies to cell margins (`w:tcMar`), table width (`w:tblW`), and paragraph spacing (`w:spacing`). Forgetting removal causes duplicate XML elements and unpredictable rendering.

### 2. TableDef Dataclass Registry

All table definitions are declared as a flat list of `TableDef` instances, separating "what to build" from "how to build it":

```python
@dataclass
class TableDef:
    csv_file: str
    label: str
    title: str
    footnotes: list[str] = field(default_factory=list)
    landscape: bool = False
```

A single `build_document()` function iterates over the list, handles missing CSVs gracefully (warns and skips), manages portrait/landscape section breaks, and calls shared helpers. Adding a new table requires only appending a `TableDef`.

### 3. Polars CSV Reading with All-String Ingestion

```python
df = pl.read_csv(filepath, infer_schema_length=0)  # all strings
```

`infer_schema_length=0` forces all columns to `Utf8`. Correct here because CSVs contain pre-formatted display values (e.g., `"0.63 (0.59-0.67)"`) that must be inserted verbatim.

### 4. python-docx Type Safety

- Use `from docx.document import Document as DocumentType` for parameter annotations
- Use `from docx import Document` for the factory constructor
- Table alignment: `WD_TABLE_ALIGNMENT.CENTER` (from `docx.enum.table`), not `WD_ALIGN_PARAGRAPH.CENTER`
- Style font access requires `# type: ignore[union-attr]` — stubs type it as `BaseStyle` but runtime is `_ParagraphStyle`

## Prevention Strategies

### Preventing language mismatch

- Document the constraint in `CLAUDE.md`: "This project uses Python exclusively"
- Code review checklist: verify new executables use Python and no new runtime dependencies outside `pyproject.toml`
- Maintain a short list of approved Python equivalents (python-docx for DOCX, openpyxl for Excel, etc.)

### Preventing gitignore drift

- Use a constant for the output directory in the Makefile (`OUTPUT_DIR := outputs`) and reference it everywhere
- When renaming any directory, grep the entire repo for the old name: `git grep -l "output/"`
- CI check: after a full run, verify `git status` shows no untracked files in outputs/

### python-docx type stubs

- Suppress Pyright only at the boundary (inline `# type: ignore` with explanation)
- Do not create custom `.pyi` stubs for python-docx — they go stale fast
- The `DocumentType` alias pattern is the cleanest approach for parameter annotations

## Cross-References

- **Related solution**: `docs/solutions/code-quality/marimo-notebook-p3-cleanup.md` (P3 cleanup patterns)
- **Plan file**: `docs/plans/2026-03-18-feat-pmfrs-vs-morse-validation-study-plan.md`
- **Dependency**: `python-docx>=1.2.0` in `pyproject.toml`
- **Makefile target**: `make docx`
