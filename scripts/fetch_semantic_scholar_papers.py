#!/usr/bin/env python3
"""Fetch TMLE-related papers from Semantic Scholar (Graph API).

This script uses unauthenticated requests by default, with optional API key
support via --api-key or SEMANTIC_SCHOLAR_API_KEY.

Output schema (JSON):
- generated_at
- source
- unauthenticated
- keywords
- paper_count
- papers: list of paper records including title/authors/abstract/year
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
API_BULK_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
DEFAULT_KEYWORDS = [
    "tmle",
    "targeted maximum likelihood estimation",
    "targeted minimum loss based estimation",
    "super learner",
    "super learning",
    "highly adaptive lasso",
]
DEFAULT_FIELDS = "paperId,title,abstract,year,publicationDate,authors,url,venue,externalIds"
DEFAULT_MIN_YEAR = 2006
DEFAULT_BLOCKLIST_JSON = "data/blocked_tmle_papers.json"


KEYWORD_PATTERNS = {
    "tmle": re.compile(r"\btmle\b", flags=re.IGNORECASE),
    "targeted maximum likelihood estimation": re.compile(
        r"targeted\s+maximum\s+likelihood\s+estimation", flags=re.IGNORECASE
    ),
    "targeted minimum loss based estimation": re.compile(
        r"targeted\s+minimum\s+loss[-\s]+based\s+estimation", flags=re.IGNORECASE
    ),
    "super learner": re.compile(r"super\s+learner", flags=re.IGNORECASE),
    "super learning": re.compile(r"super\s+learning", flags=re.IGNORECASE),
    "highly adaptive lasso": re.compile(r"highly\s+adaptive\s+lasso", flags=re.IGNORECASE),
}


@dataclass
class PaperRecord:
    title: str
    authors: List[str]
    abstract: str
    year: Optional[int]
    publication_date: Optional[str]
    paper_id: Optional[str]
    url: Optional[str]
    venue: Optional[str]
    doi: Optional[str]
    matched_keywords: List[str]
    source_queries: List[str]


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _normalize_title_for_key(title: str) -> str:
    t = _normalize_space(title).lower()
    t = re.sub(r"[^a-z0-9]+", "", t)
    return t


def _paper_key(raw: Dict) -> str:
    paper_id = raw.get("paperId")
    if paper_id:
        return f"pid:{paper_id}"

    external_ids = raw.get("externalIds") or {}
    doi = external_ids.get("DOI")
    if doi:
        return f"doi:{doi.lower()}"

    title = raw.get("title") or ""
    year = raw.get("year")
    return f"title:{_normalize_title_for_key(title)}|year:{year}"


def _record_identifiers(record: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], str]:
    """Return normalized (paper_id, doi, title+year) identifiers for persisted records."""
    paper_id = record.get("paper_id") or record.get("paperId") or record.get("id")
    if isinstance(paper_id, str):
        paper_id = _normalize_space(paper_id) or None
    else:
        paper_id = None

    doi = record.get("doi")
    if isinstance(doi, str):
        doi = _normalize_doi(doi) or None
    else:
        doi = None

    title = record.get("title") or ""
    year = record.get("year")
    title_year = f"{_normalize_title_for_key(title)}|{year}"
    return paper_id, doi, title_year


def _paper_year(raw: Dict) -> Optional[int]:
    publication_date = raw.get("publicationDate")
    if isinstance(publication_date, str):
        y = publication_date[:4]
        if y.isdigit():
            return int(y)

    year = raw.get("year")
    if isinstance(year, int):
        return year
    if isinstance(year, str) and year.isdigit():
        return int(year)
    return None


def _normalize_doi(doi: str) -> str:
    d = _normalize_space(doi).lower()
    d = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", d)
    d = re.sub(r"^doi:\s*", "", d)
    return d


def load_blocklist(path: Path) -> Dict[str, Set[str]]:
    out = {"paper_ids": set(), "dois": set(), "titles": set()}
    if not path.exists():
        return out

    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Failed to parse blocklist JSON at {path}: {exc}") from exc

    entries: List[Any] = []
    if isinstance(obj, dict):
        blocked_papers = obj.get("blocked_papers")
        if isinstance(blocked_papers, list):
            entries.extend(blocked_papers)
        for pid in obj.get("paper_ids") or []:
            entries.append({"paper_id": pid})
        for doi in obj.get("dois") or []:
            entries.append({"doi": doi})
        for title in obj.get("titles") or []:
            entries.append({"title": title})
    elif isinstance(obj, list):
        entries.extend(obj)
    else:
        raise RuntimeError(
            f"Blocklist JSON at {path} must be an object or array, got {type(obj).__name__}"
        )

    for entry in entries:
        if isinstance(entry, str):
            key = _normalize_title_for_key(entry)
            if key:
                out["titles"].add(key)
            continue
        if not isinstance(entry, dict):
            continue

        pid = (
            entry.get("paper_id")
            or entry.get("paperId")
            or entry.get("id")
        )
        if isinstance(pid, str) and _normalize_space(pid):
            out["paper_ids"].add(_normalize_space(pid))

        doi = entry.get("doi")
        if isinstance(doi, str) and _normalize_space(doi):
            out["dois"].add(_normalize_doi(doi))

        title = entry.get("title")
        if isinstance(title, str) and _normalize_space(title):
            out["titles"].add(_normalize_title_for_key(title))

    return out


def _is_blocked_raw(raw: Dict, blocked: Dict[str, Set[str]]) -> bool:
    paper_id = raw.get("paperId")
    if isinstance(paper_id, str) and _normalize_space(paper_id) in blocked["paper_ids"]:
        return True

    external_ids = raw.get("externalIds") or {}
    doi = external_ids.get("DOI") or raw.get("doi")
    if isinstance(doi, str) and _normalize_doi(doi) in blocked["dois"]:
        return True

    title_key = _normalize_title_for_key(raw.get("title") or "")
    if title_key and title_key in blocked["titles"]:
        return True

    return False


def _matched_keywords(title: str, abstract: str) -> List[str]:
    text = f"{title}\n{abstract}"
    matches = [k for k, pat in KEYWORD_PATTERNS.items() if pat.search(text)]
    return sorted(matches)


def _http_get_json(
    *,
    url: str,
    headers: Dict[str, str],
    ssl_context: Optional[ssl.SSLContext],
    timeout_s: int,
    max_retries: int,
    backoff_s: float,
    max_backoff_s: float,
) -> Dict:
    attempt = 0
    while True:
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout_s, context=ssl_context) as resp:
                body = resp.read().decode("utf-8")
            return json.loads(body)
        except urllib.error.HTTPError as exc:
            status = exc.code
            if status in (429, 500, 502, 503, 504) and attempt < max_retries:
                retry_after = exc.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    sleep_s = float(retry_after)
                else:
                    sleep_s = min(max_backoff_s, backoff_s * (2 ** attempt))
                print(
                    f"[warn] HTTP {status}; retrying in {sleep_s:.1f}s (attempt {attempt + 1}/{max_retries})",
                    file=sys.stderr,
                )
                time.sleep(sleep_s)
                attempt += 1
                continue

            err_body = ""
            try:
                err_body = exc.read().decode("utf-8")
            except Exception:
                pass
            raise RuntimeError(f"HTTP {status} for {url}\n{err_body}") from exc
        except urllib.error.URLError as exc:
            if attempt < max_retries:
                sleep_s = min(max_backoff_s, backoff_s * (2 ** attempt))
                print(
                    f"[warn] URL error {exc}; retrying in {sleep_s:.1f}s (attempt {attempt + 1}/{max_retries})",
                    file=sys.stderr,
                )
                time.sleep(sleep_s)
                attempt += 1
                continue
            raise RuntimeError(f"URL error for {url}: {exc}") from exc


def fetch_keyword(
    *,
    keyword: str,
    fields: str,
    page_size: int,
    max_results_per_keyword: int,
    request_interval_s: float,
    headers: Dict[str, str],
    ssl_context: Optional[ssl.SSLContext],
    timeout_s: int,
    max_retries: int,
    backoff_s: float,
    max_backoff_s: float,
) -> List[Dict]:
    out: List[Dict] = []
    offset = 0

    while offset < max_results_per_keyword:
        limit = min(page_size, max_results_per_keyword - offset)
        params = {
            "query": keyword,
            "offset": offset,
            "limit": limit,
            "fields": fields,
        }
        url = f"{API_URL}?{urllib.parse.urlencode(params)}"
        payload = _http_get_json(
            url=url,
            headers=headers,
            ssl_context=ssl_context,
            timeout_s=timeout_s,
            max_retries=max_retries,
            backoff_s=backoff_s,
            max_backoff_s=max_backoff_s,
        )
        page = payload.get("data") or []
        if not page:
            break
        out.extend(page)
        offset += len(page)
        print(f"[info] keyword={keyword!r} fetched={offset}", file=sys.stderr)
        if len(page) < limit:
            break
        time.sleep(request_interval_s)

    return out


def fetch_keyword_bulk(
    *,
    keyword: str,
    fields: str,
    page_size: int,
    max_results_per_keyword: int,
    request_interval_s: float,
    headers: Dict[str, str],
    ssl_context: Optional[ssl.SSLContext],
    timeout_s: int,
    max_retries: int,
    backoff_s: float,
    max_backoff_s: float,
) -> List[Dict]:
    out: List[Dict] = []
    token: Optional[str] = None

    while len(out) < max_results_per_keyword:
        limit = min(page_size, max_results_per_keyword - len(out))
        params = {
            "query": keyword,
            "limit": limit,
            "fields": fields,
        }
        if token:
            params["token"] = token
        url = f"{API_BULK_URL}?{urllib.parse.urlencode(params)}"
        payload = _http_get_json(
            url=url,
            headers=headers,
            ssl_context=ssl_context,
            timeout_s=timeout_s,
            max_retries=max_retries,
            backoff_s=backoff_s,
            max_backoff_s=max_backoff_s,
        )
        page = payload.get("data") or []
        if not page:
            break
        out.extend(page)
        print(f"[info] keyword={keyword!r} fetched={len(out)}", file=sys.stderr)

        token = payload.get("token")
        if not token:
            break
        time.sleep(request_interval_s)

    return out


def dedupe_and_filter(
    raw_papers: Iterable[Tuple[str, Dict]],
    *,
    min_year: int,
    blocked: Dict[str, Set[str]],
) -> Tuple[List[PaperRecord], int]:
    merged: Dict[str, Dict] = {}
    source_queries: Dict[str, Set[str]] = {}
    matched: Dict[str, Set[str]] = {}
    blocked_count = 0

    for query_kw, raw in raw_papers:
        if _is_blocked_raw(raw, blocked):
            blocked_count += 1
            continue

        paper_year = _paper_year(raw)
        if paper_year is None or paper_year < min_year:
            continue

        title = _normalize_space(raw.get("title") or "")
        abstract = _normalize_space(raw.get("abstract") or "")
        kws = _matched_keywords(title, abstract)
        if not kws:
            continue

        key = _paper_key(raw)
        if key not in merged:
            merged[key] = raw
            source_queries[key] = set()
            matched[key] = set()
        source_queries[key].add(query_kw)
        matched[key].update(kws)

    records: List[PaperRecord] = []
    for key, raw in merged.items():
        authors = [
            _normalize_space(a.get("name") or "")
            for a in (raw.get("authors") or [])
            if _normalize_space(a.get("name") or "")
        ]
        external_ids = raw.get("externalIds") or {}
        year_value = raw.get("year")
        if not isinstance(year_value, int):
            year_value = _paper_year(raw)

        records.append(
            PaperRecord(
                title=_normalize_space(raw.get("title") or ""),
                authors=authors,
                abstract=_normalize_space(raw.get("abstract") or ""),
                year=year_value,
                publication_date=raw.get("publicationDate"),
                paper_id=raw.get("paperId"),
                url=raw.get("url"),
                venue=_normalize_space(raw.get("venue") or "") or None,
                doi=external_ids.get("DOI"),
                matched_keywords=sorted(matched[key]),
                source_queries=sorted(source_queries[key]),
            )
        )

    records.sort(key=lambda r: (-(r.year or -9999), r.title.lower()))
    return records, blocked_count


def load_existing_output_papers(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Failed to parse existing output JSON at {path}: {exc}") from exc

    papers = obj.get("papers")
    if not isinstance(papers, list):
        raise RuntimeError(
            f"Existing output JSON at {path} has invalid 'papers' field: expected list"
        )
    return papers


def append_new_records_only(
    *,
    existing_papers: List[Dict[str, Any]],
    fetched_records: List[PaperRecord],
) -> Tuple[List[Dict[str, Any]], int, int]:
    # First, self-dedupe existing papers so append-only mode can also clean up
    # historical duplicates caused by unstable upstream paper IDs.
    merged: List[Dict[str, Any]] = []
    seen_pids: Set[str] = set()
    seen_dois: Set[str] = set()
    seen_title_years: Set[str] = set()
    removed_existing_duplicates = 0

    def is_seen(pid: Optional[str], doi: Optional[str], title_year: str) -> bool:
        # Prefer DOI as stable identity when available.
        if doi and doi in seen_dois:
            return True
        if pid and pid in seen_pids:
            return True
        if title_year and title_year in seen_title_years:
            return True
        return False

    def mark_seen(pid: Optional[str], doi: Optional[str], title_year: str) -> None:
        if pid:
            seen_pids.add(pid)
        if doi:
            seen_dois.add(doi)
        if title_year:
            seen_title_years.add(title_year)

    for paper in existing_papers:
        pid, doi, title_year = _record_identifiers(paper)
        if is_seen(pid, doi, title_year):
            removed_existing_duplicates += 1
            continue
        merged.append(paper)
        mark_seen(pid, doi, title_year)

    appended = 0
    for rec in fetched_records:
        rec_dict = asdict(rec)
        pid, doi, title_year = _record_identifiers(rec_dict)
        if is_seen(pid, doi, title_year):
            continue
        merged.append(rec_dict)
        mark_seen(pid, doi, title_year)
        appended += 1

    return merged, appended, removed_existing_duplicates


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fetch TMLE-related papers from Semantic Scholar Graph API."
    )
    p.add_argument(
        "--output",
        default="data/semantic_scholar_tmle_papers.json",
        help="Output JSON path.",
    )
    p.add_argument(
        "--keywords",
        nargs="*",
        default=DEFAULT_KEYWORDS,
        help="Keyword queries to run.",
    )
    p.add_argument(
        "--max-results-per-keyword",
        type=int,
        default=1500,
        help="Maximum number of records to fetch per keyword query.",
    )
    p.add_argument(
        "--page-size",
        type=int,
        default=1000,
        help="Page size for Semantic Scholar API pagination (bulk max is 1000).",
    )
    p.add_argument(
        "--request-interval-s",
        type=float,
        default=1.2,
        help="Sleep between successful page requests.",
    )
    p.add_argument(
        "--timeout-s",
        type=int,
        default=45,
        help="HTTP timeout per request in seconds.",
    )
    p.add_argument(
        "--max-retries",
        type=int,
        default=8,
        help="Retries per request on 429/5xx.",
    )
    p.add_argument(
        "--backoff-s",
        type=float,
        default=6.0,
        help="Base exponential backoff in seconds.",
    )
    p.add_argument(
        "--max-backoff-s",
        type=float,
        default=90.0,
        help="Maximum backoff sleep in seconds.",
    )
    p.add_argument(
        "--api-key",
        default=os.environ.get("SEMANTIC_SCHOLAR_API_KEY", ""),
        help="Optional Semantic Scholar API key.",
    )
    p.add_argument(
        "--insecure",
        action="store_true",
        help="Disable SSL certificate verification (use only if your local cert store is broken).",
    )
    p.add_argument(
        "--mode",
        choices=("bulk", "search"),
        default="bulk",
        help="Semantic Scholar endpoint mode. bulk is faster for large retrieval.",
    )
    p.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue fetching other keywords if one keyword fails.",
    )
    p.add_argument(
        "--blocked-json",
        default=DEFAULT_BLOCKLIST_JSON,
        help="JSON file listing papers to exclude (by paper_id / doi / title).",
    )
    p.add_argument(
        "--min-year",
        type=int,
        default=DEFAULT_MIN_YEAR,
        help="Only keep papers with year/publicationDate >= this year.",
    )
    p.add_argument(
        "--append-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Append only newly discovered papers to existing output JSON (default: true). "
            "Use --no-append-only to rebuild output from current fetch results."
        ),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.page_size < 1:
        raise SystemExit("--page-size must be >= 1")
    if args.mode == "search" and args.page_size > 100:
        raise SystemExit("--page-size must be in [1, 100] for search mode")
    if args.mode == "bulk" and args.page_size > 1000:
        raise SystemExit("--page-size must be in [1, 1000] for bulk mode")
    if args.min_year < 0:
        raise SystemExit("--min-year must be >= 0")

    blocked_path = Path(args.blocked_json)
    blocked = load_blocklist(blocked_path)
    print(
        "[info] blocklist loaded: "
        f"paper_ids={len(blocked['paper_ids'])}, "
        f"dois={len(blocked['dois'])}, "
        f"titles={len(blocked['titles'])}",
        file=sys.stderr,
    )

    headers = {
        "User-Agent": "tq21-semantic-scholar-fetch/1.0",
        "Accept": "application/json",
    }
    if args.api_key:
        headers["x-api-key"] = args.api_key
    ssl_context = ssl._create_unverified_context() if args.insecure else None

    all_raw: List[Tuple[str, Dict]] = []
    query_stats: List[Dict] = []
    query_errors: List[Dict] = []
    for kw in args.keywords:
        print(f"[info] fetching keyword={kw!r}", file=sys.stderr)
        try:
            if args.mode == "bulk":
                papers = fetch_keyword_bulk(
                    keyword=kw,
                    fields=DEFAULT_FIELDS,
                    page_size=args.page_size,
                    max_results_per_keyword=args.max_results_per_keyword,
                    request_interval_s=args.request_interval_s,
                    headers=headers,
                    ssl_context=ssl_context,
                    timeout_s=args.timeout_s,
                    max_retries=args.max_retries,
                    backoff_s=args.backoff_s,
                    max_backoff_s=args.max_backoff_s,
                )
            else:
                papers = fetch_keyword(
                    keyword=kw,
                    fields=DEFAULT_FIELDS,
                    page_size=args.page_size,
                    max_results_per_keyword=args.max_results_per_keyword,
                    request_interval_s=args.request_interval_s,
                    headers=headers,
                    ssl_context=ssl_context,
                    timeout_s=args.timeout_s,
                    max_retries=args.max_retries,
                    backoff_s=args.backoff_s,
                    max_backoff_s=args.max_backoff_s,
                )
        except Exception as exc:
            query_errors.append({"keyword": kw, "error": str(exc)})
            print(f"[error] keyword={kw!r} failed: {exc}", file=sys.stderr)
            if args.continue_on_error:
                continue
            raise

        query_stats.append({"keyword": kw, "fetched_raw_count": len(papers)})
        all_raw.extend((kw, p) for p in papers)

    records, blocked_match_count = dedupe_and_filter(
        all_raw,
        min_year=args.min_year,
        blocked=blocked,
    )
    out_path = Path(args.output)
    existing_papers: List[Dict[str, Any]] = []
    if args.append_only:
        existing_papers = load_existing_output_papers(out_path)
        print(
            f"[info] append-only mode enabled; existing papers={len(existing_papers)}",
            file=sys.stderr,
        )
        merged_papers, newly_appended_count, removed_existing_duplicates = append_new_records_only(
            existing_papers=existing_papers,
            fetched_records=records,
        )
    else:
        merged_papers = [asdict(r) for r in records]
        newly_appended_count = len(merged_papers)
        removed_existing_duplicates = 0

    out = {
        "generated_at": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "source": "Semantic Scholar Graph API",
        "unauthenticated": not bool(args.api_key),
        "mode": args.mode,
        "append_only": bool(args.append_only),
        "blocked_json": str(blocked_path),
        "blocked_match_count": blocked_match_count,
        "min_year": args.min_year,
        "keywords": args.keywords,
        "query_stats": query_stats,
        "query_errors": query_errors,
        "incomplete": bool(query_errors),
        "fetched_paper_count": len(records),
        "newly_appended_count": newly_appended_count,
        "existing_duplicates_removed": removed_existing_duplicates,
        "paper_count": len(merged_papers),
        "papers": merged_papers,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.append_only:
        print(
            f"[ok] wrote {len(merged_papers)} total papers to {out_path} "
            f"(appended {newly_appended_count}, fetched {len(records)}, "
            f"removed_existing_duplicates {removed_existing_duplicates})",
            file=sys.stderr,
        )
    else:
        print(f"[ok] wrote {len(merged_papers)} papers to {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
