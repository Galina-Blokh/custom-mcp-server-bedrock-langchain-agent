#!/usr/bin/env python3
"""
A/B/C search comparison script for Dataiku DSS assistant.

Compares three search tools implemented in `dataiku_tools.py`:
- Cosine similarity: search_dataiku_docs
- Heuristic (legacy): search_dataiku_docs_heuristic
- LLM re-ranker: search_dataiku_docs_llm

Usage:
  source mcp_aiagent/bin/activate
  python test_search.py \
      --queries "create dataset" "containerized execution" "Jupyter notebooks" "visual recipes vs code recipes" \
      --max-results 5 \
      --max-docs 50 \
      --force-refresh

Notes:
- The script will fetch docs (once) according to max-docs/depth and cache them in-memory.
- LLM mode requires valid AWS Bedrock credentials in .env.
- The homepage https://doc.dataiku.com/dss/latest/ is filtered from results by the tools.
"""

import argparse
import json
import sys
from typing import List

import dataiku_tools as t


def run_fetch(max_docs: int, force_refresh: bool):
    # dataiku_tools exposes the tool via LangChain's Tool API; use invoke()
    print(f"Fetching docs (max_docs={max_docs}, force_refresh={force_refresh})...", flush=True)
    res = t.fetch_dataiku_docs.invoke({"max_docs": max_docs, "force_refresh": force_refresh})
    if isinstance(res, bytes):
        res = res.decode("utf-8", errors="ignore")
    print(res)


def pretty_print(method_name: str, query: str, res_str: str):
    print(f"\n[{method_name}] Q: {query}")
    try:
        obj = json.loads(res_str) if isinstance(res_str, str) and res_str.strip().startswith('{') else {"raw": res_str}
        results = obj.get("results")
        if results is None:
            print(obj)
            return
        for i, r in enumerate(results, 1):
            url = r.get("url", "")
            score = r.get("relevance_score")
            print(f"  {i}. {url}  score={score}")
    except Exception as e:
        print(f"  [Error parsing results: {e}] Raw: {res_str[:200]}")


def run_comparison(queries: List[str], max_results: int):
    print("\n=== Cosine ===")
    for q in queries:
        res = t.search_dataiku_docs.invoke({"query": q, "max_results": max_results})
        if isinstance(res, bytes):
            res = res.decode("utf-8", errors="ignore")
        pretty_print("Cosine", q, res)

    print("\n=== Heuristic ===")
    for q in queries:
        res = t.search_dataiku_docs_heuristic.invoke({"query": q, "max_results": max_results})
        if isinstance(res, bytes):
            res = res.decode("utf-8", errors="ignore")
        pretty_print("Heuristic", q, res)

    print("\n=== LLM ===")
    for q in queries:
        res = t.search_dataiku_docs_llm.invoke({"query": q, "max_results": max_results})
        if isinstance(res, bytes):
            res = res.decode("utf-8", errors="ignore")
        pretty_print("LLM", q, res)


def run_iterative_comparison(queries: List[str], steps: int, branch: int, modes: List[str]):
    print("\n=== Iterative Guided Crawl ===")
    for q in queries:
        print(f"\n[Iterative] Q: {q}")
        for m in modes:
            res = t.iterative_expand_crawl.invoke({
                "query": q,
                "steps": steps,
                "branch": branch,
                "mode": m,
            })
            if isinstance(res, bytes):
                res = res.decode("utf-8", errors="ignore")
            try:
                obj = json.loads(res)
                top = obj.get("top_overall") or []
                print(f"  Mode: {m} ({len(top)} urls)")
                for i, r in enumerate(top, 1):
                    print(f"    {i}. {r.get('url','')}  score={r.get('relevance_score')}")
            except Exception as e:
                print(f"  Mode {m} parse error: {e}. Raw: {str(res)[:200]}")


def main():
    parser = argparse.ArgumentParser(description="Compare search tools for Dataiku DSS assistant")
    parser.add_argument("--queries", nargs="*", default=[
        "create dataset",
        "containerized execution",
        "Jupyter notebooks",
        "visual recipes vs code recipes",
        "generative ai in dataiku",
        "llm connections",
        "ai assistants",
        "vector database",
        "prompt engineering",
        "rag in dataiku",
    ], help="List of queries to test")
    parser.add_argument("--max-results", type=int, default=5, help="Top-k results per method")
    parser.add_argument("--max-docs", type=int, default=50, help="Number of docs to cache from fetch")
    parser.add_argument("--force-refresh", action="store_true", help="Force refetch docs")
    parser.add_argument("--include-iterative", action="store_true", help="Also compare iterative guided crawl modes")
    parser.add_argument("--iter-steps", type=int, default=2, help="Iterative crawl steps")
    parser.add_argument("--iter-branch", type=int, default=3, help="Iterative crawl branch factor (top-N per parent)")
    parser.add_argument(
        "--iter-modes",
        nargs="*",
        default=["cosine", "heuristic", "llm"],
        help="Iterative scoring modes to compare (cosine, heuristic, llm)"
    )

    args = parser.parse_args()

    run_fetch(max_docs=args.max_docs, force_refresh=args.force_refresh)
    run_comparison(queries=args.queries, max_results=args.max_results)
    if args.include_iterative:
        run_iterative_comparison(
            queries=args.queries,
            steps=args.iter_steps,
            branch=args.iter_branch,
            modes=args.iter_modes,
        )


if __name__ == "__main__":
    sys.exit(main())
