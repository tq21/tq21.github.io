#!/usr/bin/env python3
"""Fetch TMLE-related papers from Semantic Scholar (Graph API).

This script uses unauthenticated requests by default, with optional API key
support via --api-key or SEMANTIC_SCHOLAR_API_KEY.

Output schema (JSON):
- generated_at
- source
- unauthenticated
- keywords
- newly_appended_count
- updated_existing_count
- newly_added_papers
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
import tempfile
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


def _paper_identity_key(
    *,
    title: str,
    year: Optional[int],
    doi: Optional[str],
    paper_id: Optional[str],
) -> str:
    """Prefer publication-level identifiers over provider-specific paper ids."""
    if isinstance(doi, str) and _normalize_space(doi):
        return f"doi:{_normalize_doi(doi)}"

    title_key = _normalize_title_for_key(title)
    if title_key and year is not None:
        return f"title:{title_key}|year:{year}"

    if isinstance(paper_id, str) and _normalize_space(paper_id):
        return f"pid:{_normalize_space(paper_id)}"

    return f"title:{title_key}|year:{year}"


def _paper_key(raw: Dict) -> str:
    external_ids = raw.get("externalIds") or {}
    return _paper_identity_key(
        title=raw.get("title") or "",
        year=_paper_year(raw),
        doi=external_ids.get("DOI"),
        paper_id=raw.get("paperId"),
    )


def _paper_key_from_record(record: Dict[str, Any]) -> str:
    return _paper_identity_key(
        title=record.get("title") or "",
        year=record.get("year"),
        doi=record.get("doi"),
        paper_id=record.get("paper_id") or record.get("paperId") or record.get("id"),
    )


def _paper_year_from_record(record: Dict[str, Any]) -> Optional[int]:
    publication_date = record.get("publication_date") or record.get("publicationDate")
    if isinstance(publication_date, str):
        y = publication_date[:4]
        if y.isdigit():
            return int(y)

    year = record.get("year")
    if isinstance(year, int):
        return year
    if isinstance(year, str) and year.isdigit():
        return int(year)
    return None


def _paper_id_from_record(record: Dict[str, Any]) -> Optional[str]:
    paper_id = record.get("paper_id") or record.get("paperId") or record.get("id")
    if isinstance(paper_id, str):
        paper_id = _normalize_space(paper_id)
        if paper_id:
            return paper_id
    return None


def _paper_doi_from_record(record: Dict[str, Any]) -> Optional[str]:
    doi = record.get("doi")
    if isinstance(doi, str):
        doi = _normalize_doi(doi)
        if doi:
            return doi
    return None


def _paper_title_key_from_record(record: Dict[str, Any]) -> str:
    return _normalize_title_for_key(record.get("title") or "")


def _record_aliases(record: Dict[str, Any]) -> List[str]:
    aliases: List[str] = []
    paper_id = _paper_id_from_record(record)
    doi = _paper_doi_from_record(record)
    title_key = _paper_title_key_from_record(record)
    year = _paper_year_from_record(record)

    if paper_id:
        aliases.append(f"pid:{paper_id}")
    if doi:
        aliases.append(f"doi:{doi}")
    if title_key and year is not None:
        aliases.append(f"title-year:{title_key}|{year}")
    if title_key:
        aliases.append(f"title:{title_key}")

    return aliases


def _author_names(value: Any) -> List[str]:
    out: List[str] = []
    for author in value or []:
        if isinstance(author, dict):
            name = _normalize_space(author.get("name") or "")
        else:
            name = _normalize_space(str(author))
        if name:
            out.append(name)
    return out


def _paper_quality_key(record: Dict[str, Any]) -> Tuple[int, int, int, int, int, int, int, int, int]:
    year = _paper_year_from_record(record) or -9999
    abstract = _normalize_space(record.get("abstract") or "")
    publication_date = record.get("publication_date") or record.get("publicationDate") or ""
    venue = _normalize_space(record.get("venue") or "")
    doi = _paper_doi_from_record(record)
    matched_keywords = record.get("matched_keywords") or []
    source_queries = record.get("source_queries") or []
    authors = _author_names(record.get("authors"))
    return (
        year,
        1 if isinstance(doi, str) and _normalize_space(doi) else 0,
        1 if publication_date else 0,
        1 if venue else 0,
        1 if abstract else 0,
        len(abstract),
        len(matched_keywords),
        len(source_queries),
        sum(len(a) for a in authors),
    )


def _prefer_better_record(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    return right if _paper_quality_key(right) > _paper_quality_key(left) else left


def _paper_summary(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "title": _normalize_space(record.get("title") or ""),
        "year": _paper_year_from_record(record),
        "authors": _author_names(record.get("authors")),
        "publication_date": record.get("publication_date") or record.get("publicationDate"),
        "doi": _paper_doi_from_record(record),
        "paper_id": _paper_id_from_record(record),
    }


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


def _is_blocked_record(record: Dict[str, Any], blocked: Dict[str, Set[str]]) -> bool:
    paper_id = _paper_id_from_record(record)
    if paper_id and paper_id in blocked["paper_ids"]:
        return True

    doi = _paper_doi_from_record(record)
    if doi and doi in blocked["dois"]:
        return True

    title_key = _paper_title_key_from_record(record)
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
        except (urllib.error.URLError, TimeoutError) as exc:
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


@dataclass
class QueryResult:
    papers: List[Dict]
    pages_fetched: int
    estimated_total: Optional[int]
    truncated: bool = False
    truncation_reason: Optional[str] = None

    def stats(self, keyword: str) -> Dict[str, Any]:
        return {
            "keyword": keyword,
            "fetched_raw_count": len(self.papers),
            "pages_fetched": self.pages_fetched,
            "estimated_total": self.estimated_total,
            "truncated": self.truncated,
            "truncation_reason": self.truncation_reason,
        }


def _response_page(payload: Dict) -> List[Dict]:
    # An error or malformed response must not masquerade as an empty result set.
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise RuntimeError("Search API response must contain a 'data' array")
    page = payload["data"]
    for paper in page:
        if not isinstance(paper, dict):
            raise RuntimeError("Search API 'data' entries must be objects")
        if not isinstance(paper.get("paperId"), str) or not paper["paperId"].strip():
            raise RuntimeError("Search API paper is missing a valid paperId")
        if not isinstance(paper.get("title"), str):
            raise RuntimeError("Search API paper is missing a valid title")
        for field in ("abstract", "publicationDate", "url", "venue"):
            if paper.get(field) is not None and not isinstance(paper[field], str):
                raise RuntimeError(f"Search API paper has invalid {field}")
        year = paper.get("year")
        if year is not None and not (type(year) is int or isinstance(year, str) and year.isdigit()):
            raise RuntimeError("Search API paper has invalid year")
        authors = paper.get("authors")
        if authors is not None and (
            not isinstance(authors, list) or any(
                not isinstance(author, dict) or
                author.get("name") is not None and not isinstance(author["name"], str)
                for author in authors
            )
        ):
            raise RuntimeError("Search API paper has invalid authors")
        external_ids = paper.get("externalIds")
        if external_ids is not None and (
            not isinstance(external_ids, dict) or
            external_ids.get("DOI") is not None and not isinstance(external_ids["DOI"], str)
        ):
            raise RuntimeError("Search API paper has invalid externalIds")
    return page


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
) -> QueryResult:
    # The relevance endpoint exposes at most 1,000 results. Bulk is the default
    # and the only suitable mode when a query has more matches.
    cap = min(max_results_per_keyword or 1000, 1000)
    result = QueryResult([], 0, None)
    offset = 0
    while offset < cap:
        limit = min(page_size, cap - offset)
        params = {"query": keyword, "offset": offset, "limit": limit, "fields": fields}
        payload = _http_get_json(
            url=f"{API_URL}?{urllib.parse.urlencode(params)}",
            headers=headers, ssl_context=ssl_context, timeout_s=timeout_s,
            max_retries=max_retries, backoff_s=backoff_s, max_backoff_s=max_backoff_s,
        )
        page = _response_page(payload)
        result.pages_fetched += 1
        if isinstance(payload.get("total"), int):
            result.estimated_total = payload["total"]
        next_offset = payload.get("next")
        more = next_offset is not None or (
            result.estimated_total is not None and result.estimated_total > offset + len(page)
        )
        if not page:
            if more:
                raise RuntimeError("Relevance API returned an empty page before exhaustion")
            return result
        remaining = cap - len(result.papers)
        result.papers.extend(page[:remaining])
        print(f"[info] keyword={keyword!r} fetched={len(result.papers)}", file=sys.stderr)
        if len(page) > remaining or (len(result.papers) >= cap and more):
            result.truncated = True
            result.truncation_reason = "Relevance search result limit reached; use bulk mode"
            return result
        if not more:
            return result
        if type(next_offset) is not int or next_offset != offset + len(page):
            raise RuntimeError("Relevance API returned an invalid or missing continuation offset")
        offset = next_offset
        time.sleep(request_interval_s)
    return result


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
) -> QueryResult:
    result = QueryResult([], 0, None)
    token: Optional[str] = None
    seen_tokens: Set[str] = set()
    while True:
        # Bulk controls its own batch size (up to 1,000) and has no 'limit'
        # parameter. Follow every token, including short pages and undated papers.
        params = {"query": keyword, "fields": fields, "sort": "publicationDate:desc"}
        if token:
            params["token"] = token
        payload = _http_get_json(
            url=f"{API_BULK_URL}?{urllib.parse.urlencode(params)}",
            headers=headers, ssl_context=ssl_context, timeout_s=timeout_s,
            max_retries=max_retries, backoff_s=backoff_s, max_backoff_s=max_backoff_s,
        )
        page = _response_page(payload)
        result.pages_fetched += 1
        if isinstance(payload.get("total"), int):
            result.estimated_total = payload["total"]
        next_token = payload.get("token")
        if next_token is not None and not isinstance(next_token, str):
            raise RuntimeError("Bulk API returned an invalid continuation token")
        if not page and next_token:
            raise RuntimeError("Bulk API returned an empty page with a continuation token")
        remaining = max_results_per_keyword - len(result.papers) if max_results_per_keyword else len(page)
        result.papers.extend(page[:remaining])
        print(f"[info] keyword={keyword!r} fetched={len(result.papers)}", file=sys.stderr)
        if len(page) > remaining or (
            max_results_per_keyword and len(result.papers) >= max_results_per_keyword and next_token
        ):
            result.truncated = True
            result.truncation_reason = "Configured result cap reached before query exhaustion"
            return result
        if not next_token:
            return result
        if next_token in seen_tokens:
            raise RuntimeError("Bulk API repeated a continuation token; stopping pagination loop")
        seen_tokens.add(next_token)
        token = next_token
        time.sleep(request_interval_s)


def write_json_atomic(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary_path = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


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
        else:
            merged[key] = _prefer_better_record(merged[key], raw)
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


def _collapse_record_group(
    papers: List[Dict[str, Any]],
    key_fn,
) -> List[Dict[str, Any]]:
    order: List[str] = []
    best_by_key: Dict[str, Dict[str, Any]] = {}
    passthrough: List[Dict[str, Any]] = []

    for paper in papers:
        key = key_fn(paper)
        if not key:
            passthrough.append(paper)
            continue
        if key not in best_by_key:
            order.append(key)
            best_by_key[key] = paper
            continue
        best_by_key[key] = _prefer_better_record(best_by_key[key], paper)

    return passthrough + [best_by_key[key] for key in order]


def dedupe_persisted_papers(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped = list(papers)
    deduped = _collapse_record_group(
        deduped,
        lambda p: f"pid:{_paper_id_from_record(p)}" if _paper_id_from_record(p) else None,
    )
    deduped = _collapse_record_group(
        deduped,
        lambda p: f"doi:{_paper_doi_from_record(p)}" if _paper_doi_from_record(p) else None,
    )
    deduped = _collapse_record_group(
        deduped,
        lambda p: (
            f"title-year:{_paper_title_key_from_record(p)}|{_paper_year_from_record(p)}"
            if _paper_title_key_from_record(p) and _paper_year_from_record(p) is not None
            else None
        ),
    )
    # Treat same-title records across years as version chains and keep the most recent.
    deduped = _collapse_record_group(
        deduped,
        lambda p: f"title:{_paper_title_key_from_record(p)}" if _paper_title_key_from_record(p) else None,
    )
    deduped.sort(
        key=lambda p: (
            -(_paper_year_from_record(p) or -9999),
            _normalize_space(p.get("title") or "").lower(),
        )
    )
    return deduped


def append_new_records_only(
    *,
    existing_papers: List[Dict[str, Any]],
    fetched_records: List[PaperRecord],
) -> Tuple[List[Dict[str, Any]], int, int, int, List[Dict[str, Any]]]:
    merged: List[Dict[str, Any]] = dedupe_persisted_papers(existing_papers)
    removed_existing_duplicates = max(0, len(existing_papers) - len(merged))
    alias_to_index: Dict[str, int] = {}
    for idx, paper in enumerate(merged):
        for alias in _record_aliases(paper):
            alias_to_index[alias] = idx

    appended = 0
    updated_existing = 0
    added_records: List[Dict[str, Any]] = []
    for rec in fetched_records:
        rec_dict = asdict(rec)
        aliases = _record_aliases(rec_dict)
        matched_indexes = {alias_to_index[alias] for alias in aliases if alias in alias_to_index}
        if matched_indexes:
            idx = min(matched_indexes)
            old_record = merged[idx]
            merged[idx] = _prefer_better_record(old_record, rec_dict)
            if merged[idx] is not old_record:
                updated_existing += 1
                for alias in _record_aliases(merged[idx]):
                    alias_to_index[alias] = idx
            continue
        merged.append(rec_dict)
        idx = len(merged) - 1
        for alias in aliases:
            alias_to_index[alias] = idx
        appended += 1
        added_records.append(_paper_summary(rec_dict))

    merged = dedupe_persisted_papers(merged)
    return merged, appended, updated_existing, removed_existing_duplicates, added_records


def filter_blocked_persisted_papers(
    papers: List[Dict[str, Any]],
    blocked: Dict[str, Set[str]],
) -> Tuple[List[Dict[str, Any]], int]:
    kept: List[Dict[str, Any]] = []
    removed = 0
    for paper in papers:
        if _is_blocked_record(paper, blocked):
            removed += 1
            continue
        kept.append(paper)
    return kept, removed


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
        "--report",
        help="Optional diagnostic JSON report, written on success or query failure without full paper records.",
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
        default=0,
        help="Maximum raw records per keyword; 0 exhausts all bulk pages (default). A reached cap fails the run.",
    )
    p.add_argument(
        "--page-size",
        type=int,
        default=1000,
        help="Page size for relevance search (max 100). Bulk uses server-sized batches up to 1000.",
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
        help="Check remaining keywords after an error; still exit nonzero and preserve existing output if any fail.",
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
    if args.max_results_per_keyword < 0:
        raise SystemExit("--max-results-per-keyword must be >= 0")
    if not args.keywords or any(not kw.strip() for kw in args.keywords):
        raise SystemExit("At least one nonempty --keywords query is required")
    if args.request_interval_s < 0 or args.max_retries < 0:
        raise SystemExit("Request interval and retry count must be nonnegative")
    if args.timeout_s <= 0 or args.backoff_s <= 0 or args.max_backoff_s <= 0:
        raise SystemExit("Timeout and backoff values must be positive")
    if args.report and Path(args.report).resolve() == Path(args.output).resolve():
        raise SystemExit("--report and --output must be different paths")

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
    query_truncations: List[Dict] = []
    for index, kw in enumerate(args.keywords):
        if index:
            time.sleep(args.request_interval_s)
        print(f"[info] fetching keyword={kw!r}", file=sys.stderr)
        try:
            if args.mode == "bulk":
                result = fetch_keyword_bulk(
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
                result = fetch_keyword(
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
            break

        query_stats.append(result.stats(kw))
        if result.truncated:
            query_truncations.append({"keyword": kw, "reason": result.truncation_reason})
        all_raw.extend((kw, p) for p in result.papers)

    if query_errors or query_truncations:
        report = {
            "generated_at": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "source": "Semantic Scholar Graph API",
            "mode": args.mode,
            "keywords": args.keywords,
            "query_stats": query_stats,
            "query_errors": query_errors,
            "query_truncations": query_truncations,
            "incomplete": True,
            "output_updated": False,
        }
        if args.report:
            try:
                write_json_atomic(Path(args.report), report)
            except OSError as exc:
                print(f"[error] Could not save diagnostic report: {exc}", file=sys.stderr)
        print("[error] Discovery incomplete; existing output was not changed. "
              "Resolve query errors or remove the result cap before generating the page.", file=sys.stderr)
        return 2

    records, blocked_match_count = dedupe_and_filter(
        all_raw,
        min_year=args.min_year,
        blocked=blocked,
    )
    out_path = Path(args.output)
    existing_papers: List[Dict[str, Any]] = []
    removed_existing_blocked = 0
    updated_existing_count = 0
    newly_added_papers: List[Dict[str, Any]] = []
    if args.append_only:
        existing_papers = load_existing_output_papers(out_path)
        existing_papers, removed_existing_blocked = filter_blocked_persisted_papers(
            existing_papers,
            blocked,
        )
        print(
            f"[info] append-only mode enabled; existing papers={len(existing_papers)} "
            f"(removed_blocked={removed_existing_blocked})",
            file=sys.stderr,
        )
        (
            merged_papers,
            newly_appended_count,
            updated_existing_count,
            removed_existing_duplicates,
            newly_added_papers,
        ) = append_new_records_only(
            existing_papers=existing_papers,
            fetched_records=records,
        )
    else:
        merged_papers = dedupe_persisted_papers([asdict(r) for r in records])
        newly_appended_count = len(merged_papers)
        updated_existing_count = 0
        removed_existing_duplicates = 0
        newly_added_papers = [_paper_summary(p) for p in merged_papers]

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
        "query_truncations": query_truncations,
        "incomplete": False,
        "sort": "publicationDate:desc" if args.mode == "bulk" else "relevance",
        "max_results_per_keyword": args.max_results_per_keyword,
        "fetched_paper_count": len(records),
        "newly_appended_count": newly_appended_count,
        "updated_existing_count": updated_existing_count,
        "existing_blocked_removed": removed_existing_blocked,
        "existing_duplicates_removed": removed_existing_duplicates,
        "newly_added_papers": newly_added_papers,
        "paper_count": len(merged_papers),
        "papers": merged_papers,
    }

    # The report is required when requested. Write it before replacing a good
    # catalog so an unwritable report destination cannot cause an unreported update.
    report = {**{k: v for k, v in out.items() if k != "papers"}, "output_updated": False}
    try:
        if args.report:
            write_json_atomic(Path(args.report), report)
        write_json_atomic(out_path, out)
    except OSError as exc:
        print(f"[error] Could not save discovery output: {exc}", file=sys.stderr)
        return 2
    if args.report:
        report["output_updated"] = True
        try:
            write_json_atomic(Path(args.report), report)
        except OSError as exc:
            # The first diagnostic report is already durable; the catalog is
            # complete, but the runner must not continue if final reporting fails.
            print(f"[error] Catalog saved, but could not finalize diagnostic report: {exc}", file=sys.stderr)
            return 2
    if args.append_only:
        print(
            f"[ok] wrote {len(merged_papers)} total papers to {out_path} "
            f"(appended {newly_appended_count}, fetched {len(records)}, "
            f"updated_existing {updated_existing_count}, "
            f"removed_existing_blocked {removed_existing_blocked}, "
            f"removed_existing_duplicates {removed_existing_duplicates})",
            file=sys.stderr,
        )
    else:
        print(f"[ok] wrote {len(merged_papers)} papers to {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
