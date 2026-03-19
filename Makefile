.PHONY: setup sync check lint format edit run export-html export-pdf docx clean help

# ─── Configuration ────────────────────────────────────────────────────
NOTEBOOKS := $(wildcard notebooks/*.py)
HTML_DIR  := outputs/html
PDF_DIR   := outputs/pdf

# ─── Setup ────────────────────────────────────────────────────────────
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup: ## First-time project setup (install uv, deps, pre-commit)
	@command -v uv >/dev/null 2>&1 || { echo "Installing uv..."; curl -LsSf https://astral.sh/uv/install.sh | sh; }
	uv sync
	uv run pre-commit install
	@echo "\n✓ Project ready. Run 'make edit' to start."

sync: ## Install/update all dependencies from lockfile
	uv sync

lock: ## Regenerate uv.lock (after editing pyproject.toml)
	uv lock

upgrade: ## Upgrade all dependencies to latest compatible versions
	uv lock --upgrade
	uv sync

# ─── Development ──────────────────────────────────────────────────────
edit: ## Open marimo editor (pass NB=notebooks/foo.py to pick notebook)
	uv run marimo edit $(or $(NB),.)

run: ## Run a notebook as a script (pass NB=notebooks/foo.py)
	@test -n "$(NB)" || { echo "Usage: make run NB=notebooks/foo.py"; exit 1; }
	uv run $(NB)

run-all: ## Run all notebooks as scripts (CI smoke test)
	@for nb in $(NOTEBOOKS); do \
		echo "── Running $$nb ──"; \
		uv run $$nb || exit 1; \
	done

# ─── Code Quality ────────────────────────────────────────────────────
check: ## Run marimo structural checks on all notebooks
	@for nb in $(NOTEBOOKS); do \
		echo "── Checking $$nb ──"; \
		uvx marimo check $$nb || exit 1; \
	done

lint: ## Run ruff linter
	uv run ruff check .

format: ## Run ruff formatter
	uv run ruff format .

typecheck: ## Run pyright type checker
	uv run pyright notebooks/ scripts/

quality: lint check typecheck ## Run all quality checks (lint + marimo check + typecheck)

# ─── Export ───────────────────────────────────────────────────────────
export-html: ## Export all notebooks to HTML
	@mkdir -p $(HTML_DIR)
	@for nb in $(NOTEBOOKS); do \
		name=$$(basename $$nb .py); \
		echo "── Exporting $$name.html ──"; \
		uv run marimo export html $$nb -o $(HTML_DIR)/$$name.html || exit 1; \
	done
	@echo "\n✓ HTML exports in $(HTML_DIR)/"

export-md: ## Export all notebooks to Markdown
	@mkdir -p outputs/md
	@for nb in $(NOTEBOOKS); do \
		name=$$(basename $$nb .py); \
		echo "── Exporting $$name.md ──"; \
		uv run marimo export md $$nb -o outputs/md/$$name.md || exit 1; \
	done

docx: ## Generate JAMA-formatted DOCX tables from CSVs
	uv run scripts/generate_docx_tables.py
	@echo "\n✓ DOCX files in outputs/docx/"

# ─── Cleanup ──────────────────────────────────────────────────────────
clean: ## Remove build artifacts and caches
	rm -rf outputs/ __pycache__ .ruff_cache .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# ─── Data ─────────────────────────────────────────────────────────────
data-check: ## Verify data files exist and show basic info
	@echo "Data files:"
	@ls -lh data/ 2>/dev/null || echo "  No data/ directory yet"
	@echo ""
	@echo "Excel files in project root:"
	@ls -lh *.xlsx 2>/dev/null || echo "  None"
