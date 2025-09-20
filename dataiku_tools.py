import asyncio
import json

from langchain_community.document_loaders import RecursiveUrlLoader
from bs4 import BeautifulSoup as Soup
from langchain.tools import tool
import re
from collections import Counter
import math
import os
from dotenv import load_dotenv
from langchain_aws import ChatBedrockConverse
from urllib.parse import urljoin
from urllib.request import Request, urlopen

# Shared in-memory cache
cached_docs = None
cached_urls = set()

# Ensure environment is loaded when running tools standalone (outside agent)
load_dotenv()

# Lightweight LLM client for optional LLM-based re-ranking
try:
    _MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0")
    _REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    _AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    _AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    _AWS_SESSION_TOKEN = os.getenv("AWS_SESSION_TOKEN")
    llm = ChatBedrockConverse(
        model=_MODEL_ID,
        region_name=_REGION,
        aws_access_key_id=_AWS_ACCESS_KEY_ID,
        aws_secret_access_key=_AWS_SECRET_ACCESS_KEY,
        temperature=0.1,
        max_tokens=700,
        verbose=False,
    )
except Exception:
    llm = None


@tool
def fetch_dataiku_docs(max_docs: int = 50, force_refresh: bool = False, continue_fetching: bool = False) -> str:
    """Fetch Dataiku documentation from https://doc.dataiku.com/dss/latest/.
    Set continue_fetching=True to expand the knowledge base from previous fetches.
    """
    global cached_docs, cached_urls

    # If we already have a full in-memory corpus, always use it (skip crawling)
    if cached_docs is not None and len(cached_docs) >= 544:
        return (
            f"Using cached full documentation ({len(cached_docs)} documents). "
            f"No crawl needed in current session."
        )

    # Return cached docs if available and not forcing refresh or continuing
    if cached_docs is not None and not force_refresh and not continue_fetching:
        return (
            f"Using cached documentation ({len(cached_docs)} documents). "
            f"Use force_refresh=true to reload or continue_fetching=true to expand."
        )

    try:
        # Custom extractor function to handle different content types
        def custom_extractor(content):
            try:
                soup = Soup(content, 'html.parser')

                # Remove script and style elements
                for script in soup(["script", "style", "nav", "footer", "header"]):
                    script.decompose()

                # Get text content
                text = soup.get_text()

                # Clean up whitespace
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = ' '.join(chunk for chunk in chunks if chunk)

                return text
            except Exception as e:
                return f"Error parsing content: {str(e)}"

        # Load documents using RecursiveUrlLoader with optimized settings
        loader = RecursiveUrlLoader(
            "https://doc.dataiku.com/dss/latest/",
            prevent_outside=True,
            use_async=True,
            timeout=45,  # Adjust as needed
            max_depth=2,  # Adjust as needed
            extractor=custom_extractor,
            exclude_dirs=[
                "_downloads", "_static", "_sources", "_images",
                "troubleshooting", "release_notes", "preparation/processors",
                "thirdparty", "plugins/reference"
            ]
        )

        if continue_fetching and cached_docs is not None:
            print(f"Continuing to fetch Dataiku documentation (current: {len(cached_docs)} documents)...")
            # additional_docs = max_docs * 2  # noqa: F841 (kept for potential future logic)
        else:
            print("Starting to fetch Dataiku documentation...")
            # additional_docs = max_docs  # noqa: F841

        docs = loader.load()
        print(f"Fetched {len(docs)} documents")

        # Handle incremental fetching
        if continue_fetching and cached_docs is not None:
            # Merge with existing cached docs
            existing_urls = cached_urls
            new_docs = [doc for doc in docs if doc.metadata.get('source', '') not in existing_urls]
            combined_docs = cached_docs + new_docs
            print(f"Added {len(new_docs)} new documents to existing {len(cached_docs)} documents")
            docs = combined_docs
        else:
            # Limit the number of documents for initial fetch
            if len(docs) > max_docs:
                docs = docs[:max_docs]

        # Cache the results
        cached_docs = docs
        cached_urls = {doc.metadata.get('source', '') for doc in docs if doc.metadata.get('source')}

        # Extract URLs for response
        urls = [doc.metadata.get('source', '') for doc in docs if doc.metadata.get('source')]

        response = {
            "status": "success",
            "documents_loaded": len(docs),
            "urls_found": len(urls),
            "sample_urls": urls[:10]
        }

        return json.dumps(response, indent=2)

    except Exception as e:
        return f"Error fetching docs: {str(e)}"


def _tokenize(text: str):
    # Lowercase, keep alphanumerics, split on non-word chars
    return [t for t in re.findall(r"\b\w+\b", text.lower()) if t]

def _cosine_similarity(a_tokens, b_tokens) -> float:
    if not a_tokens or not b_tokens:
        return 0.0
    a_counts = Counter(a_tokens)
    b_counts = Counter(b_tokens)
    # Dot product
    dot = sum(a_counts[t] * b_counts.get(t, 0) for t in a_counts)
    if dot == 0:
        return 0.0
    # Norms
    norm_a = math.sqrt(sum(v*v for v in a_counts.values()))
    norm_b = math.sqrt(sum(v*v for v in b_counts.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

def _is_homepage(url: str) -> bool:
    if not url:
        return False
    u = url.strip()
    if u.endswith('index.html'):
        u = u[:-10]
    return u.rstrip('/') == 'https://doc.dataiku.com/dss/latest'

_DOC_BASE = 'https://doc.dataiku.com/dss/latest/'
_EXCLUDE_DIRS = {
    "_downloads", "_static", "_sources", "_images",
    "troubleshooting", "release_notes", "preparation/processors",
    "thirdparty", "plugins/reference"
}

def _is_allowed(url: str) -> bool:
    if not url:
        return False
    try:
        # Strip fragments
        url = _strip_fragment(url)
        if not url.startswith(_DOC_BASE):
            return False
        # Exclude directories
        lower = url.lower()
        for d in _EXCLUDE_DIRS:
            if f"/{d}/" in lower:
                return False
        return True
    except Exception:
        return False

def _fetch_html(url: str, timeout: int = 45) -> str:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read().decode('utf-8', errors='ignore')

def _strip_fragment(url: str) -> str:
    try:
        return url.split('#', 1)[0]
    except Exception:
        return url

def _extract_links(html: str, base_url: str) -> list:
    # Use BeautifulSoup parser via Soup alias
    try:
        soup = Soup(html, 'html.parser')
    except Exception:
        return []
    links = []
    seen = set()
    for a in soup.find_all('a', href=True):
        href = a.get('href')
        abs_url = _strip_fragment(urljoin(base_url, href))
        if abs_url in seen:
            continue
        if _is_allowed(abs_url) and not _is_homepage(abs_url):
            anchor_text = (a.get_text() or '').strip()
            links.append((abs_url, anchor_text))
            seen.add(abs_url)
    return links

@tool
def search_dataiku_docs_heuristic(query: str, max_results: int = 5) -> str:
    """Search cached docs using heuristic scoring (legacy method)."""
    global cached_docs

    if cached_docs is None:
        return "No documents loaded. Please call fetch_dataiku_docs first."

    if not query:
        return "Please provide a search query."

    try:
        # Search through documents
        results = []
        query_words = query.lower().split()

        for doc in cached_docs:
            content = doc.page_content.lower()
            source = doc.metadata.get('source', '')

            # Calculate relevance score with improved logic
            relevance_score = 0

            # Exact phrase match gets highest score
            if query.lower() in content:
                relevance_score += 20

            # Boost score for exact word matches in content
            for word in query_words:
                word_count = content.count(word)
                relevance_score += word_count * 2

            # Title/page name relevance - heavily weighted
            if source:
                page_name = source.split('/')[-1].replace('.html', '').replace('-', ' ').replace('_', ' ')
                for word in query_words:
                    if word in page_name.lower():
                        relevance_score += 15

            # Penalize release notes and other non-instructional content
            if any(term in source.lower() for term in ['release_notes', 'changelog', 'version']):
                relevance_score -= 10

            # Boost score for instructional/tutorial content
            if any(term in source.lower() for term in ['tutorial', 'guide', 'how-to', 'getting-started', 'first-']):
                relevance_score += 10

            # Boost score for main topic pages
            if any(term in source.lower() for term in ['dataset', 'preparation', 'flow', 'recipe']):
                relevance_score += 8

            if relevance_score > 0:
                # Extract relevant snippet
                words = content.split()
                best_match = ""
                best_score = 0

                # Find the best matching section
                for i, word in enumerate(words):
                    if any(qw in word for qw in query_words):
                        start = max(0, i - 30)
                        end = min(len(words), i + 30)
                        snippet = " ".join(words[start:end])
                        snippet_score = sum(1 for qw in query_words if qw in snippet.lower())
                        if snippet_score > best_score:
                            best_score = snippet_score
                            best_match = snippet

                results.append({
                    "url": source,
                    "snippet": best_match[:300] + "..." if len(best_match) > 300 else best_match,
                    "relevance_score": relevance_score
                })

        # Filter out release notes and other non-instructional content
        filtered_results = []
        for result in results:
            source = result['url'].lower()
            # Skip release notes, changelogs, and version pages
            if (
                not any(term in source for term in ['release_notes', 'changelog', 'version', 'thirdparty', 'plugins/reference'])
                and not _is_homepage(result['url'])
            ):
                filtered_results.append(result)

        # Sort by relevance (raw heuristic) and limit results
        filtered_results.sort(key=lambda x: x['relevance_score'], reverse=True)

        # If we don't have enough good results, try a broader search
        if len(filtered_results) < max_results and len(filtered_results) < 3:
            # Try searching with individual keywords
            for word in query_words:
                if len(word) > 3:  # Only for meaningful words
                    for doc in cached_docs:
                        content = doc.page_content.lower()
                        source = doc.metadata.get('source', '')

                        # Skip if already in results or is release notes
                        if any(result['url'] == source for result in filtered_results):
                            continue
                        if any(term in source.lower() for term in ['release_notes', 'changelog', 'version']):
                            continue

                        if word in content and word in source.lower():
                            # Extract relevant snippet
                            words = content.split()
                            best_match = ""
                            for i, w in enumerate(words):
                                if word in w:
                                    start = max(0, i - 20)
                                    end = min(len(words), i + 20)
                                    snippet = " ".join(words[start:end])
                                    if len(snippet) > len(best_match):
                                        best_match = snippet

                            filtered_results.append({
                                "url": source,
                                "snippet": best_match[:300] + "..." if len(best_match) > 300 else best_match,
                                "relevance_score": 5
                            })

        # Take top-k then normalize scores to 0..100 relative to the max in this list
        results = filtered_results[:max_results]
        if results:
            max_score = max(r['relevance_score'] for r in results) or 1
            for r in results:
                r['relevance_score'] = int(round((r['relevance_score'] / max_score) * 100))

        response = {
            "query": query,
            "results_found": len(results),
            "results": results
        }

        return json.dumps(response, indent=2)

    except Exception as e:
        return f"Error searching docs: {str(e)}"


@tool
def search_dataiku_docs_llm(query: str, max_results: int = 5) -> str:
    """Re-rank cached docs with an LLM without using cosine preselection.
    Candidates are selected using a lightweight heuristic (keyword presence
    and simple boosts/penalties), then the LLM chooses the top URLs.
    Returns JSON with results and normalized 0..100 relevance scores.
    """
    global cached_docs

    if cached_docs is None:
        return "No documents loaded. Please call fetch_dataiku_docs first."

    if not query:
        return "Please provide a search query."

    if llm is None:
        return "LLM client not available. Ensure AWS credentials and Bedrock access are configured."

    try:
        # Preselect candidates with a lightweight heuristic (no cosine)
        q_words = [w for w in _tokenize(query) if len(w) > 1]
        qset = set(q_words)
        candidates = []
        for doc in cached_docs:
            content = doc.page_content or ""
            source = doc.metadata.get('source', '')
            if not source or _is_homepage(source):
                continue
            lower_src = source.lower()
            if any(term in lower_src for term in ['release_notes', 'changelog', 'version', 'thirdparty', 'plugins/reference']):
                continue

            title = source.split('/')[-1].replace('.html', '').replace('-', ' ').replace('_', ' ')
            title_tokens = _tokenize(title)
            content_lower = content.lower()

            # Simple heuristic: token overlap + boosts + penalties
            score = 0
            # Count query words in title and content
            title_overlap = sum(1 for t in title_tokens if t in qset)
            score += title_overlap * 5  # title match is strong
            for w in q_words:
                # count occurrences in content (cheap substring count)
                cnt = content_lower.count(w)
                if cnt:
                    score += min(cnt, 5)  # cap contribution per word

            # Boost instructional content
            if any(k in lower_src for k in ['tutorial', 'guide', 'how-to', 'getting-started', 'first-']):
                score += 5
            # Penalize non-instructional already filtered mostly

            if score > 0:
                words = content.split()
                snippet = " ".join(words[:80]) if words else ""
                candidates.append({
                    "url": source,
                    "title": title,
                    "base": float(score),  # keep heuristic base for later normalization
                    "snippet": snippet[:400]
                })

        # Keep top 20 candidates by heuristic to limit prompt size
        candidates.sort(key=lambda x: x['base'], reverse=True)
        candidates = candidates[:20]
        if not candidates:
            return json.dumps({"query": query, "results_found": 0, "results": []}, indent=2)

        # Build prompt for LLM re-ranking (LLM must judge relevance strictly)
        lines = []
        for i, c in enumerate(candidates, 1):
            lines.append(
                f"{i}. URL: {c['url']}\n"
                f"   Title: {c['title']}\n"
                f"   Snippet: {c['snippet']}"
            )
        prompt = (
            "You are validating and ranking documentation pages STRICTLY by relevance to the user query.\n"
            "Instructions:\n"
            "- Prefer pages that directly address the task or concept asked in the query.\n"
            "- Prefer tutorials, how-to, guides, getting-started pages when applicable.\n"
            "- Avoid generic hubs, high-level indices, release notes, changelogs, or weakly related topics.\n"
            "- If a page is only tangentially related, do not include it.\n"
            "- Provide a SHORT reason for each selected URL explaining why it is relevant.\n\n"
            "Given the query and the candidate pages (URL, title, and snippet), return the top results as strict JSON only.\n"
            "Schema: {\"results\": [{\"url\": \"...\", \"reason\": \"...\"}, ...]}\n\n"
            f"Query: {query}\n\nCandidates:\n" + "\n\n".join(lines) +
            f"\n\nReturn the best {max_results} unique URLs as JSON only, no extra text."
        )

        ai = llm.invoke(prompt)
        text = getattr(ai, 'content', ai)
        # Extract JSON
        start = text.find('{')
        end = text.rfind('}')
        if start == -1 or end == -1:
            raise ValueError('LLM did not return JSON')
        data = json.loads(text[start:end+1])

        # Build results list with homepage filtered (already filtered) and limit count
        out = []
        seen = set()
        # Rank-based scoring so that LLM's ordering defines relevance (no numeric pre-scores)
        for item in data.get('results', []):
            url = item.get('url')
            if not url or url in seen or _is_homepage(url):
                continue
            seen.add(url)
            # Rank-based 0..100 score: 100, 90, 80, ... (minimum 10)
            rel = max(100 - len(out) * 10, 10)
            out.append({
                "url": url,
                "snippet": next((c['snippet'] for c in candidates if c['url'] == url), ""),
                "relevance_score": rel
            })
            if len(out) >= max_results:
                break

        return json.dumps({
            "query": query,
            "results_found": len(out),
            "results": out
        }, indent=2)

    except Exception as e:
        return f"Error in LLM search: {str(e)}"


@tool
def iterative_expand_crawl(query: str, steps: int = 2, branch: int = 3, per_request_timeout: int = 45, mode: str = "cosine") -> str:
    """Guided iterative crawl starting from the homepage based on query relevance.

    At step 0, start from the homepage and select top-N outgoing links (branch) by cosine similarity
    (using URL title + anchor text). For each subsequent step, expand each selected page by picking
    its top-N outgoing links. Accumulates unique URLs and returns per-step selections.

    Returns JSON with per-step selections and a flattened unique URL list. Homepage is never included.
    """
    global cached_docs, cached_urls

    try:
        frontier = [_DOC_BASE]
        visited = set()
        per_step = []
        all_urls = set()
        overall_scores = {}  # url -> base score (cosine in [0,1] or heuristic raw)

        q_tokens = _tokenize(query)

        mode = (mode or "cosine").lower()

        def score_heuristic(url: str, anchor: str) -> float:
            # Simple heuristic using tokens from title+anchor and URL hints
            title = url.split('/')[-1].replace('.html', '').replace('-', ' ').replace('_', ' ')
            tokens = _tokenize(title + ' ' + (anchor or ''))
            qset = set(q_tokens)
            score = 0
            # token overlap
            score += sum(1 for t in tokens if t in qset)
            # URL hints
            u = url.lower()
            if any(k in u for k in ['dataset', 'preparation', 'flow', 'recipe']):
                score += 2
            if any(k in u for k in ['tutorial', 'guide', 'how-to', 'getting-started', 'first-']):
                score += 2
            if any(k in u for k in ['release_notes', 'changelog', 'version', 'thirdparty', 'plugins/reference']):
                score -= 3
            return float(max(score, 0))

        def llm_pick_top(parent_url: str, cand: list, k: int) -> list:
            """Use LLM to pick top-k URLs for this parent. cand is list of dicts with url,title,anchor,base.
            Returns list of URLs.
            """
            if llm is None or not cand:
                return [c['url'] for c in sorted(cand, key=lambda x: x.get('base', 0), reverse=True)[:k]]
            lines = []
            for i, c in enumerate(cand, 1):
                lines.append(f"{i}. URL: {c['url']}\n   Title: {c['title']}\n   Anchor: {c['anchor']}\n   BaseScore: {c['base']:.4f}")
            prompt = (
                "You are ranking outgoing links by relevance to a user query. "
                "Given the query and candidates from a parent documentation page, return the top URLs as JSON.\n"
                "Schema: {\"results\": [{\"url\": \"...\"}]}\n\n"
                f"Query: {query}\nParent: {parent_url}\n\nCandidates:\n" + "\n\n".join(lines) +
                f"\n\nReturn the best {k} unique URLs as JSON only."
            )
            try:
                ai = llm.invoke(prompt)
                text = getattr(ai, 'content', ai)
                start = text.find('{'); end = text.rfind('}')
                if start == -1 or end == -1:
                    raise ValueError('LLM did not return JSON')
                data = json.loads(text[start:end+1])
                urls = []
                seen = set()
                for item in data.get('results', []):
                    u = item.get('url')
                    if u and u not in seen:
                        urls.append(u); seen.add(u)
                    if len(urls) >= k:
                        break
                if not urls:
                    # fallback to base score
                    urls = [c['url'] for c in sorted(cand, key=lambda x: x.get('base', 0), reverse=True)[:k]]
                return urls
            except Exception:
                return [c['url'] for c in sorted(cand, key=lambda x: x.get('base', 0), reverse=True)[:k]]

        for step in range(steps):
            next_frontier = []
            step_selected = []
            for parent in frontier:
                if parent in visited:
                    continue
                visited.add(parent)
                # Fetch parent page
                try:
                    html = _fetch_html(parent, timeout=per_request_timeout)
                except Exception:
                    continue
                # Extract links
                links = _extract_links(html, parent)
                # Build candidates with base score (cosine or heuristic)
                candidates = []
                for url, anchor in links:
                    if url in all_urls:
                        continue
                    title = url.split('/')[-1].replace('.html', '').replace('-', ' ').replace('_', ' ')
                    if mode == 'heuristic':
                        base = score_heuristic(url, anchor)
                    else:
                        tokens = _tokenize(title + ' ' + (anchor or ''))
                        base = _cosine_similarity(q_tokens, tokens)
                    if base > 0:
                        candidates.append({"url": url, "title": title, "anchor": anchor, "base": base})
                        # Track best base score overall for final top-5
                        if url not in overall_scores or base > overall_scores[url]:
                            overall_scores[url] = base

                # Select top-k for this parent based on mode
                if mode == 'llm':
                    selected = llm_pick_top(parent, candidates, branch)
                else:
                    selected = [c['url'] for c in sorted(candidates, key=lambda x: x['base'], reverse=True)[:branch]]
                # Ensure uniqueness and not homepage
                selected = [u for u in selected if not _is_homepage(u)]
                step_selected.extend(selected)
                # add to next frontier
                for u in selected:
                    if u not in all_urls:
                        next_frontier.append(u)
                        all_urls.add(u)

            per_step.append({
                "step": step + 1,
                "selected_count": len(step_selected),
                "urls": step_selected
            })
            frontier = next_frontier
            if not frontier:
                break

        # Build overall top-5 list across all discovered URLs
        overall_list = [
            {"url": u, "_base": s}
            for u, s in overall_scores.items() if not _is_homepage(u)
        ]
        overall_list.sort(key=lambda x: x["_base"], reverse=True)
        overall_list = overall_list[:5]
        # Normalize to 0..100 depending on mode
        if mode == 'heuristic':
            if overall_list:
                max_s = max(item["_base"] for item in overall_list) or 1
                for item in overall_list:
                    item["relevance_score"] = int(round((item["_base"] / max_s) * 100))
        else:
            for item in overall_list:
                item["relevance_score"] = int(round(item["_base"] * 100))
        for item in overall_list:
            item.pop("_base", None)

        response = {
            "query": query,
            "steps": steps,
            "branch": branch,
            "total_urls": len(all_urls),
            "per_step": per_step,
            "unique_urls": list(all_urls)[:200],
            "top_overall": overall_list
        }
        return json.dumps(response, indent=2)

    except Exception as e:
        return f"Error in iterative crawl: {str(e)}"


@tool
def search_dataiku_docs(query: str, max_results: int = 5) -> str:
    """Search cached docs using cosine similarity on tokens (content + title),
    favoring instructional pages and filtering out non-instructional ones.
    """
    global cached_docs

    if cached_docs is None:
        return "No documents loaded. Please call fetch_dataiku_docs first."

    if not query:
        return "Please provide a search query."

    try:
        q_tokens = _tokenize(query)
        results = []

        for doc in cached_docs:
            content = doc.page_content or ""
            source = doc.metadata.get('source', '')

            # Build tokens for content and title
            content_tokens = _tokenize(content)
            title = ''
            if source:
                title = source.split('/')[-1].replace('.html', '').replace('-', ' ').replace('_', ' ')
            title_tokens = _tokenize(title)

            # Compute cosine similarities
            sim_content = _cosine_similarity(q_tokens, content_tokens)
            sim_title = _cosine_similarity(q_tokens, title_tokens)

            # Weighted combined similarity (title gets a strong weight)
            combined = 0.6 * sim_title + 0.4 * sim_content

            # Penalize known non-instructional pages
            if any(term in source.lower() for term in ['release_notes', 'changelog', 'version', 'thirdparty', 'plugins/reference']):
                combined *= 0.5

            if combined > 0:
                # Extract a snippet around the first matching token occurrence
                words = (doc.page_content or '').split()
                best_match = ""
                if words:
                    # Find index of first occurrence of any query token
                    qset = set(q_tokens)
                    idx = next((i for i, w in enumerate(words) if any(t in w.lower() for t in qset)), None)
                    if idx is not None:
                        start = max(0, idx - 30)
                        end = min(len(words), idx + 30)
                        best_match = " ".join(words[start:end])

                results.append({
                    "url": source,
                    "snippet": (best_match[:300] + "...") if len(best_match) > 300 else best_match,
                    "_combined": combined  # keep raw for sorting; expose normalized later
                })

        # Remove homepage results explicitly
        results = [r for r in results if not _is_homepage(r.get('url', ''))]
        # Sort by combined similarity desc and take top-k
        results.sort(key=lambda x: x.get('_combined', 0.0), reverse=True)
        results = results[:max_results]
        # Normalize to 0..100 and drop internal fields
        for r in results:
            r['relevance_score'] = int(round((r.get('_combined', 0.0)) * 100))
            r.pop('_combined', None)

        response = {
            "query": query,
            "results_found": len(results),
            "results": results
        }

        return json.dumps(response, indent=2)

    except Exception as e:
        return f"Error searching docs: {str(e)}"


@tool
def get_doc_urls(filter_pattern: str = "") -> str:
    """Get list of available Dataiku documentation URLs"""
    global cached_urls

    if not cached_urls:
        return "No URLs available. Please call fetch_dataiku_docs first."

    # Filter URLs if pattern provided
    filtered_urls = list(cached_urls)
    if filter_pattern:
        filtered_urls = [url for url in cached_urls if filter_pattern.lower() in url.lower()]

    response = {
        "total_urls": len(cached_urls),
        "filtered_urls": len(filtered_urls),
        "urls": sorted(filtered_urls)
    }

    return json.dumps(response, indent=2)


async def prefetch_docs(initial_max_docs: int = 50, depth: int = 1, force_refresh: bool = False) -> None:
    """Prefetch documentation in the background using a worker thread."""
    global cached_docs, cached_urls
    if cached_docs is not None and not force_refresh:
        return

    def _load_docs_sync():
        def custom_extractor(content):
            try:
                soup = Soup(content, 'html.parser')
                for script in soup(["script", "style", "nav", "footer", "header"]):
                    script.decompose()
                text = soup.get_text()
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                return ' '.join(chunk for chunk in chunks if chunk)
            except Exception as e:
                return f"Error parsing content: {str(e)}"

        loader = RecursiveUrlLoader(
            "https://doc.dataiku.com/dss/latest/",
            prevent_outside=True,
            use_async=True,
            timeout=60,
            max_depth=depth,
            extractor=custom_extractor,
            exclude_dirs=[
                "_downloads", "_static", "_sources", "_images",
                "troubleshooting", "release_notes", "preparation/processors",
                "thirdparty", "plugins/reference"
            ]
        )
        docs_local = loader.load()
        if len(docs_local) > initial_max_docs:
            docs_local = docs_local[:initial_max_docs]
        urls_local = {doc.metadata.get('source', '') for doc in docs_local if doc.metadata.get('source')}
        return docs_local, urls_local

    try:
        docs_local, urls_local = await asyncio.to_thread(_load_docs_sync)
        cached_docs = docs_local
        cached_urls = urls_local
        # print(f"Background prefetch complete: {len(cached_docs)} documents cached")
    except Exception as e:
        print(f"Background prefetch failed: {e}")
