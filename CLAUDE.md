# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

simpllm is a unified Python library for interacting with multiple LLM providers (Anthropic Claude and Google Gemini). It provides a stateless, type-safe interface with support for extended thinking, tool calling, prompt caching, and structured outputs.

## Development Commands

The project includes a Makefile for common development tasks:

### Setup
```bash
make install          # Install dependencies including dev tools
```

### Testing
```bash
make test             # Run all tests
make test-unit        # Run only unit tests
make test-integration # Run integration tests (requires GOOGLE_API_KEY)
```

To run a specific test:
```bash
uv run pytest tests/test_basic.py::test_user_message_from_text -v
```

### Code Quality
```bash
make lint             # Run ruff linter
make format           # Format code with ruff
make typecheck        # Run mypy type checker
make check            # Run lint + typecheck + unit tests
```

### Building and Publishing
```bash
make build            # Build distribution packages
make clean            # Clean build artifacts
make publish-test     # Publish to TestPyPI
make publish          # Publish to PyPI (requires PYPI_TOKEN)
```

### Manual Commands
```bash
uv sync --extra dev   # Install dependencies
uv run pytest tests/  # Run all tests
uv run ruff check src/ tests/
uv run ruff format src/ tests/
uv run mypy src/
uv build              # Build package
```

## Architecture

### Core Design Principles

1. **Stateless Wrappers**: The wrapper classes (`AnthropicWrapper`, `GeminiWrapper`) do NOT maintain conversation state. Callers must pass the full message history to `invoke()` each time.

2. **Unified Message Format**: All provider-specific message formats are converted to/from a common message structure:
   - `UserMessage` and `AssistantMessage` with typed content blocks
   - Content blocks: `UserTextBlock`, `AssistantTextBlock`, `ThinkingBlock`, `ToolCallBlock`, `ToolResultBlock`
   - This abstraction allows switching providers without changing application code

3. **Type Safety**: The entire library uses Pydantic models for validation and serialization. Generic type parameters ensure type safety across provider implementations.

### Key Components

**messages.py**: Defines the unified message format
- Content blocks are discriminated unions for type safety
- `AssistantMessage.to_aggregated_response()` merges all text/thinking/tool_calls into a single response object
- `UserMessage.from_text()` and `UserMessage.from_tool_results()` are convenience constructors

**wrappers.py**: Provider implementations
- `ProviderWrapper[Client, Message, Response, ToolDeclaration]` is the generic abstract base class
- Each wrapper implements conversion between native provider types and unified message format
- `invoke()` is the main entry point - it's stateless and returns an `AssistantMessage`
- Wrappers handle retries with exponential backoff via tenacity

**tools.py**: Tool/function calling support
- `BaseTool` is a Pydantic model with an `invoke(context)` method
- `__tool_name__` class attribute is auto-set from class name via `__init_subclass__`
- Can be customized: `class MyTool(BaseTool, tool_name="custom_name")`

**utils.py**: Schema conversion utilities
- `pydantic_to_function_declaration()` converts `BaseTool` subclasses to provider-specific tool schemas
- Handles Gemini's `parameters` format and Anthropic's `input_schema` format
- Uses `GenerateJsonSchemaNoTitles` to clean up schemas (removes unnecessary title fields)

### Provider-Specific Details

**Anthropic Wrapper**:
- Supports prompt caching (controlled by `use_cache` flag)
- Thinking budget and interleaved thinking support
- Token usage includes cache_read_input_tokens and cache_creation_input_tokens

**Gemini Wrapper**:
- Structured output support via `output_schema` parameter (Pydantic model)
- Returns parsed response in `AssistantMessage.structured_output`
- Default thinking_budget is -1 (unlimited)

## Common Patterns

### Message Handling
When building conversation history, alternate between `UserMessage` and `AssistantMessage`:
```python
messages = [
    UserMessage.from_text("Hello"),
    assistant_response,  # From previous invoke()
    UserMessage.from_text("Follow-up question"),
]
response = await wrapper.invoke(messages)
```

### Tool Results
After LLM requests tool calls, convert tool results back to messages:
```python
aggregated = response.to_aggregated_response()
if aggregated.tool_calls:
    tool_results = [execute_tool(tc) for tc in aggregated.tool_calls]
    messages.append(response)  # Add assistant message with tool calls
    messages.append(UserMessage.from_tool_results(tool_results))
    response = await wrapper.invoke(messages)  # Continue conversation
```

### Token Counting
Before making expensive requests:
```python
token_count = await wrapper.count_input_tokens(messages)
if token_count > threshold:
    # Truncate or summarize
```

## Environment Configuration

The library expects these environment variables:
- `ANTHROPIC_API_KEY`: For Claude models (can be customized via `api_key_env_var` parameter)
- `GOOGLE_API_KEY`: For Gemini models

A `.env` file is present in the project root (gitignored).

## CI/CD and Release Process

### GitHub Actions Workflows

**CI Workflow** (`.github/workflows/ci.yml`):
- Runs on push to master/main and pull requests
- Tests against Python 3.12 and 3.13
- Runs linting (ruff), formatting checks, and type checking (mypy)
- Executes unit tests on all commits
- Runs integration tests on push (if `GOOGLE_API_KEY` secret is set)

**Lint Workflow** (`.github/workflows/lint.yml`):
- Runs on pull requests only
- Fast feedback for code quality checks

**Release Workflow** (`.github/workflows/release.yml`):
- Triggers on version tags (e.g., `v0.1.3`)
- Runs tests, builds package
- Creates GitHub release with auto-generated notes
- Publishes to PyPI using GitHub's trusted publisher (OIDC)

### Creating a Release

1. Update version in `pyproject.toml`
2. Commit changes: `git commit -am "Bump version to X.Y.Z"`
3. Create and push tag: `git tag vX.Y.Z && git push origin vX.Y.Z`
4. GitHub Actions will automatically build and publish to PyPI

### Required Secrets

Configure these in GitHub repository settings → Secrets:
- `GOOGLE_API_KEY`: Google API key for integration tests (optional)

Note: PyPI publishing uses GitHub's trusted publisher feature (OIDC), so no PyPI token is needed.

## Testing Notes

**Unit Tests** (`tests/test_basic.py`):
- Message creation and conversion
- Tool declaration generation
- Wrapper initialization with/without tools
- Response aggregation
- Do NOT make actual API calls - only verify schema conversions and object initialization

**Integration Tests** (`tests/test_integration.py`):
- Make actual API calls to Google Gemini (requires `GOOGLE_API_KEY` environment variable)
- Use `gemini-2.5-flash` model for testing
- Test basic text generation, tool calling, token counting, and multi-turn conversations
- Tests are skipped automatically if `GOOGLE_API_KEY` is not set
- Note: Thinking features require paid tier, so thinking tests are commented out for free tier usage