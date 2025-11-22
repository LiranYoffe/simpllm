.PHONY: test lint format typecheck install clean build

install:
	uv sync --extra dev

test:
	uv run pytest tests/

test-unit:
	uv run pytest tests/test_basic.py -v

test-integration:
	uv run pytest tests/test_integration.py -v

lint:
	uv run ruff check src/ tests/

format:
	uv run ruff format src/ tests/
	uv run ruff check --fix src/ tests/

typecheck:
	uv run mypy src/

check: lint typecheck test-unit
	@echo "All checks passed!"

build:
	uv build

clean:
	rm -rf dist/ build/ *.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +

publish-test:
	uv publish --publish-url https://test.pypi.org/legacy/

publish:
	uv publish
