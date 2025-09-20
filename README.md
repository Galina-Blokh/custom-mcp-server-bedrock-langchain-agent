# Dataiku DSS Documentation Assistant

This project provides a custom Dataiku DSS documentation assistant using LangChain's RecursiveUrlLoader to fetch and search documentation content.

## Quick Start

Run the agent (LLM mode by default):

```bash
source mcp_aiagent/bin/activate
python agent_simple_custom.py --iter-mode llm
```

Run the test comparison (cosine, heuristic, LLM):

```bash
source mcp_aiagent/bin/activate
python test_search.py --max-docs 50 --max-results 3 --force-refresh
```

Include iterative guided crawl comparisons:

```bash
python test_search.py \
  --queries "create dataset" "containerized execution" \
  --max-docs 50 --max-results 3 \
  --include-iterative --iter-steps 2 --iter-branch 3 --iter-modes llm heuristic cosine \
  --force-refresh
```

## Files

### Application
- **`agent_simple_custom.py`** — Interactive CLI agent (only entrypoint)

### Testing
- **`test_search.py`** — Compares three search tools (cosine, heuristic, LLM re-ranker)

### Other
- **`__init__.py`** - Python package initialization
- **`mcp_aiagent/`** - Virtual environment with all required dependencies

## Usage

### Run the Agent (interactive)
```bash
source mcp_aiagent/bin/activate
# Choose iterative mode at startup: llm (default) | heuristic | cosine
python agent_simple_custom.py --iter-mode llm
```

Behavior:
- The agent performs an iterative, relevance-guided expansion per query using `iterative_expand_crawl(steps=2, branch=3, mode="<your-mode>")`.
- It returns only the `top_overall` URLs from the iterative crawl (no backfill). If fewer than 5 are found, it returns fewer and states that explicitly.
- Homepage and fragment URLs are filtered.

### Run the Search Comparison Test
This script fetches docs once, then runs the three tools side-by-side and prints the top results. You can also include iterative guided crawl comparisons across modes.

```bash
source mcp_aiagent/bin/activate
python test_search.py \
  --queries "create dataset" "containerized execution" "Jupyter notebooks" "visual recipes vs code recipes" \
  --max-results 5 \
  --max-docs 50 \
  --force-refresh
```

Include iterative guided crawl (optional):
```bash
python test_search.py \
  --queries "create dataset" "containerized execution" \
  --max-results 3 \
  --max-docs 50 \
  --include-iterative \
  --iter-steps 2 \
  --iter-branch 3 \
  --iter-modes cosine heuristic llm \
  --force-refresh
```

Notes:
- The homepage `https://doc.dataiku.com/dss/latest/` is filtered out by the tools.
- LLM re-ranker requires valid AWS Bedrock credentials in `.env`.
- Fetch behavior (depth/timeout/excludes) is configured in `dataiku_tools.py`.
- Test output lists each method, query, URL, and the normalized relevance score (0–100).
- Fragment URLs (with `#...`) are ignored during crawling and ranking.
- Iterative guided crawl (`iterative_expand_crawl`) defaults to `steps=2`, `branch=3` and returns `top_overall` (up to 5 URLs). The agent does not backfill—if fewer are found, it returns fewer.
- Agent iterative mode can be selected at startup with `--iter-mode {llm|heuristic|cosine}`. The default is `llm`.
- LLM checker (`search_dataiku_docs_llm`) judges relevance directly from URL/title/snippet (no cosine preselection). Scores are rank-based (100, 90, 80, ...).

## Features

- **Custom Document Loading**: Uses RecursiveUrlLoader to fetch Dataiku documentation
- **Intelligent Caching**: Caches documents for better performance
- **Smart Search**: Built-in search with relevance scoring
- **URL Filtering**: Can filter URLs by patterns for specific topics
- **Error Handling**: Gracefully handles timeout errors and problematic URLs

## Requirements

- Python 3.10+
- Virtual environment with required packages (see `mcp_aiagent/`)
- AWS credentials for Bedrock (in `.env` file)

## How It Works

1. **Document Fetching**: Uses RecursiveUrlLoader to crawl https://doc.dataiku.com/dss/latest/
2. **Content Processing**: Extracts clean text from HTML using BeautifulSoup
3. **Caching**: Stores fetched documents in memory for performance
4. **Search**: Provides intelligent search through cached documentation
5. **Response**: Returns relevant URLs and content snippets based on queries

The custom implementation provides better control, caching, and search capabilities compared to the standard mcp-server-fetch tool.
