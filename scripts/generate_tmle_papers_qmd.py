#!/usr/bin/env python3
"""Generate tmle-papers.qmd from Semantic Scholar JSON data."""

from __future__ import annotations

import argparse
import html
import json
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


DEFAULT_BLOCKLIST_JSON = "data/blocked_tmle_papers.json"


SURNAME_PARTICLES = {
    "da",
    "de",
    "del",
    "della",
    "der",
    "di",
    "du",
    "la",
    "le",
    "van",
    "von",
    "den",
    "ten",
    "ter",
    "st",
    "saint",
    "bin",
    "ibn",
    "al",
}


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def normalize_title_for_key(title: str) -> str:
    t = fold_ascii(normalize_space(title)).lower()
    return re.sub(r"[^a-z0-9]+", "", t)


def normalize_doi(doi: str) -> str:
    d = normalize_space(doi).lower()
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
            key = normalize_title_for_key(entry)
            if key:
                out["titles"].add(key)
            continue
        if not isinstance(entry, dict):
            continue

        pid = entry.get("paper_id") or entry.get("paperId") or entry.get("id")
        if isinstance(pid, str):
            pid = normalize_space(pid)
            if pid:
                out["paper_ids"].add(pid)

        doi = entry.get("doi")
        if isinstance(doi, str):
            doi = normalize_space(doi)
            if doi:
                out["dois"].add(normalize_doi(doi))

        title = entry.get("title")
        if isinstance(title, str):
            title = normalize_space(title)
            if title:
                out["titles"].add(normalize_title_for_key(title))

    return out


def is_blocked_paper(paper: Dict, blocked: Dict[str, Set[str]]) -> bool:
    paper_id = paper.get("paper_id")
    if isinstance(paper_id, str) and normalize_space(paper_id) in blocked["paper_ids"]:
        return True

    doi = paper.get("doi")
    if isinstance(doi, str) and normalize_doi(doi) in blocked["dois"]:
        return True

    title = paper.get("title")
    if isinstance(title, str) and normalize_title_for_key(title) in blocked["titles"]:
        return True

    return False


def filter_blocked_papers(
    papers: List[Dict], blocked: Dict[str, Set[str]]
) -> Tuple[List[Dict], int]:
    kept: List[Dict] = []
    blocked_count = 0
    for paper in papers:
        if is_blocked_paper(paper, blocked):
            blocked_count += 1
            continue
        kept.append(paper)
    return kept, blocked_count


def parse_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except Exception:
        return None


def month_start(d: date) -> date:
    return date(d.year, d.month, 1)


def next_month(d: date) -> date:
    if d.month == 12:
        return date(d.year + 1, 1, 1)
    return date(d.year, d.month + 1, 1)


def fold_ascii(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def parse_author_name(name: str) -> Tuple[str, bool, str]:
    """Return (first_token, is_initial, surname_phrase) for matching."""
    s = fold_ascii(normalize_space(name))
    tokens = re.findall(r"[A-Za-z]+\.?", s)
    if not tokens:
        return ("", False, "")
    toks = [t.lower() for t in tokens]

    first = toks[0].rstrip(".")
    is_initial = len(first) == 1

    i = len(toks) - 1
    surname = [toks[i].rstrip(".")]
    i -= 1
    while i >= 1:
        tok = toks[i].rstrip(".")
        if tok in SURNAME_PARTICLES:
            surname.insert(0, tok)
            i -= 1
            continue
        break
    surname_phrase = " ".join(surname)
    return (first, is_initial, surname_phrase)


def display_name_score(name: str) -> Tuple[int, int]:
    parts = normalize_space(name).split()
    has_middle = 1 if len(parts) >= 3 else 0
    return (has_middle, len(name))


def label_quality(name: str) -> Tuple[int, int, int, int, int]:
    """Higher is better for display labels.

    Preference order:
    1) non-initial first name (full first name)
    2) more non-initial tokens (more complete spelling)
    3) any non-initial middle token
    4) more tokens
    5) longer string
    """
    tokens = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]+\.?", normalize_space(name))
    if not tokens:
        return (0, 0, 0, 0, 0)

    def is_non_initial(tok: str) -> bool:
        return len(tok.strip(".")) > 1

    first_full = 1 if is_non_initial(tokens[0]) else 0
    non_initial_count = sum(1 for t in tokens if is_non_initial(t))
    middle_full = 1 if any(is_non_initial(t) for t in tokens[1:-1]) else 0
    return (first_full, non_initial_count, middle_full, len(tokens), len(name))


def forced_author_key(first: str, surname: str) -> Optional[str]:
    # Explicit alias handling for Mark van der Laan variants.
    # Covers forms like "Mark ... van der Laan", "M. J. van der Laan",
    # and shorthand "M. J. Laan".
    fi = (first or "")[:1]
    if fi == "m" and surname in {"van der laan", "laan"}:
        return "van der laan|mark"
    return None


def build_monthly_chart(papers: List[Dict]) -> str:
    start_year = 2006
    end_year = date.today().year - 1

    def resolve_year(p: Dict) -> Optional[int]:
        pd = parse_date(p.get("publication_date"))
        if pd:
            return pd.year
        y = p.get("year")
        if isinstance(y, int):
            return y
        return None

    year_vals = [resolve_year(p) for p in papers]
    in_range = [y for y in year_vals if y is not None and start_year <= y <= end_year]
    if not in_range:
        return "\n".join(
            [
                "## Yearly Papers Trend",
                "",
                f"Year metadata in the **{start_year}-{end_year}** range is not available for these records.",
            ]
        )

    counts: Dict[int, int] = {}
    for y in in_range:
        counts[y] = counts.get(y, 0) + 1

    years = list(range(start_year, end_year + 1))
    series = [(y, counts.get(y, 0)) for y in years]
    max_y = max(v for _, v in series)

    step_x = 34
    left = 56
    right = 18
    top = 16
    plot_h = 220
    bottom = 44
    plot_w = max(1, (len(series) - 1) * step_x)
    svg_w = left + plot_w + right
    svg_h = top + plot_h + bottom

    if max_y <= 6:
        ticks = list(range(0, max_y + 1))
    else:
        step = max(1, (max_y + 4) // 5)
        ticks = list(range(0, max_y + 1, step))
        if ticks[-1] != max_y:
            ticks.append(max_y)

    def x_for(i: int) -> float:
        return left + i * step_x

    def y_for(v: int) -> float:
        if max_y == 0:
            return top + plot_h
        return top + plot_h - (v / max_y) * plot_h

    line_pts = [(x_for(i), y_for(v)) for i, (_, v) in enumerate(series)]
    line_str = " ".join(f"{x:.2f},{y:.2f}" for x, y in line_pts)
    area_str = (
        f"{left:.2f},{top+plot_h:.2f} "
        + line_str
        + f" {x_for(len(series)-1):.2f},{top+plot_h:.2f}"
    )

    x_ticks: List[Tuple[int, str]] = [(i, str(y)) for i, (y, _) in enumerate(series)]

    lines: List[str] = []
    lines.append("## Yearly Papers Trend")
    lines.append("")
    lines.append(
        f"Number of papers per year from **{start_year}** to **{end_year}** "
        f"(coverage: **{len(in_range)}/{len(papers)}** papers with year metadata in range)."
    )
    lines.append("")
    lines.append("```{=html}")
    lines.append("<style>")
    lines.append(".yearly-chart-wrap { overflow-x: auto; border: 1px solid #d7dde3; border-radius: 8px; padding: 8px; background: #fff; }")
    lines.append(".yearly-chart-svg { display: block; width: auto; max-width: none; height: auto; }")
    lines.append(".yearly-chart-svg text { fill: #374151; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; font-size: 10px; }")
    lines.append(".yearly-chart-note { color: #4b5563; font-size: 0.92rem; margin-top: 0.4rem; }")
    lines.append("</style>")
    lines.append('<div class="yearly-chart-wrap">')
    lines.append(
        f'<svg class="yearly-chart-svg" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {svg_w} {svg_h}" '
        f'width="{svg_w}" height="{svg_h}" role="img" aria-label="Papers per year line chart">'
    )

    for t in ticks:
        y = y_for(t)
        lines.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left+plot_w}" y2="{y:.2f}" stroke="#e5e7eb" stroke-width="1"/>'
        )
        lines.append(f'<text x="{left-8}" y="{y+3:.2f}" text-anchor="end">{t}</text>')

    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#374151" stroke-width="1.2"/>')
    lines.append(
        f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#374151" stroke-width="1.2"/>'
    )
    lines.append(f'<polygon points="{area_str}" fill="#93c5fd" opacity="0.26"/>')
    lines.append(f'<polyline points="{line_str}" fill="none" stroke="#1d4ed8" stroke-width="2"/>')

    for i, (year_lbl, v) in enumerate(series):
        x = x_for(i)
        y_coord = y_for(v)
        tip = html.escape(f"{year_lbl}: {v} papers")
        lines.append(f'<circle cx="{x:.2f}" cy="{y_coord:.2f}" r="2.2" fill="#1d4ed8"><title>{tip}</title></circle>')

    for i, lbl in x_ticks:
        x = x_for(i)
        lines.append(
            f'<line x1="{x:.2f}" y1="{top+plot_h}" x2="{x:.2f}" y2="{top+plot_h+4}" stroke="#374151" stroke-width="1"/>'
        )
        lines.append(f'<text x="{x:.2f}" y="{top+plot_h+16}" text-anchor="middle">{lbl}</text>')

    y_mid = top + plot_h / 2
    lines.append(
        f'<text x="18" y="{y_mid:.2f}" transform="rotate(-90 18 {y_mid:.2f})" text-anchor="middle">Papers / year</text>'
    )
    lines.append("</svg>")
    lines.append("</div>")
    lines.append('<div class="yearly-chart-note">Tip: hover points for exact counts.</div>')
    lines.append("```")
    return "\n".join(lines)


def build_contributor_chart(papers: List[Dict]) -> str:
    mentions: List[Tuple[str, str, bool, str]] = []
    full_keys_by_sig: Dict[Tuple[str, str], set] = defaultdict(set)

    for p in papers:
        for author in p.get("authors") or []:
            full = normalize_space(author)
            if not full:
                continue
            first, is_initial, surname = parse_author_name(full)
            if not first or not surname:
                continue
            mentions.append((full, first, is_initial, surname))
            if not is_initial:
                key = forced_author_key(first, surname) or f"{surname}|{first}"
                full_keys_by_sig[(surname, first[0])].add(key)

    counts = Counter()
    label_candidates = defaultdict(Counter)
    for full, first, is_initial, surname in mentions:
        forced = forced_author_key(first, surname)
        if forced:
            key = forced
        elif is_initial:
            candidates = full_keys_by_sig.get((surname, first[0]), set())
            if len(candidates) == 1:
                key = next(iter(candidates))
            else:
                key = f"{surname}|{first}."
        else:
            key = f"{surname}|{first}"
        counts[key] += 1
        label_candidates[key][full] += 1

    labels: Dict[str, str] = {}
    for key, cand in label_candidates.items():
        # If any full-name form exists, always prefer it over initial-only forms.
        full_pool = [n for n in cand if label_quality(n)[0] == 1]
        pool = full_pool if full_pool else list(cand.keys())
        best = sorted(pool, key=lambda n: label_quality(n) + (cand[n],), reverse=True)[0]
        labels[key] = best

    if "van der laan|mark" in labels:
        labels["van der laan|mark"] = "Mark van der Laan (the godfather himself)"

    rows = [(labels[k], v) for k, v in counts.items() if v >= 2]
    rows.sort(key=lambda x: (-x[1], x[0]))
    if not rows:
        return "## Contributor Ranking\n\nNo contributors with at least 2 papers found."

    max_count = rows[0][1]
    plot_height = 260

    lines: List[str] = []
    lines.append("## Contributor Ranking")
    lines.append("")
    lines.append(
        f"Bars show contributors with at least 2 papers, "
        f"sorted left to right by paper count. Total shown: **{len(rows)}**."
    )
    lines.append("")
    lines.append("```{=html}")
    lines.append("<style>")
    lines.append(".contrib-bars-wrap { overflow-x: auto; border: 1px solid #d7dde3; border-radius: 8px; padding: 10px; background: #fff; }")
    lines.append(".contrib-bars { display: flex; align-items: flex-end; gap: 4px; min-width: max-content; height: 470px; }")
    lines.append(".contrib-item { width: 14px; flex: 0 0 14px; display: flex; flex-direction: column; align-items: center; justify-content: flex-end; }")
    lines.append(".contrib-count { width: 0; height: 10px; position: relative; overflow: visible; margin-bottom: 2px; }")
    lines.append(".contrib-count span { position: absolute; left: 50%; transform: translateX(-50%); white-space: nowrap; font-size: 8px; color: #4b5563; line-height: 1; pointer-events: none; }")
    lines.append(".contrib-bar { width: 12px; background: #1f77b4; border-radius: 3px 3px 0 0; min-height: 4px; }")
    lines.append(".contrib-item.top .contrib-bar { background: #0f5f9a; }")
    lines.append(".contrib-label { margin-top: 2px; width: 0; height: 180px; position: relative; overflow: visible; }")
    lines.append(".contrib-label span { position: absolute; left: 0; top: 0; transform-origin: left top; transform: rotate(30deg); white-space: nowrap; font-size: 10px; color: #111827; line-height: 1; pointer-events: none; }")
    lines.append(".contrib-note { color: #4b5563; font-size: 0.92rem; margin-top: 0.45rem; }")
    lines.append("</style>")
    lines.append('<div class="contrib-bars-wrap">')
    lines.append('  <div class="contrib-bars">')

    for i, (name, val) in enumerate(rows):
        h = max(4.0, (val / max_count) * plot_height)
        cls = "contrib-item top" if i == 0 else "contrib-item"
        esc_name = html.escape(name)
        title = html.escape(f"{name}: {val} papers")
        lines.append(
            f'    <div class="{cls}" title="{title}"><div class="contrib-count"><span>{val}</span></div>'
            f'<div class="contrib-bar" style="height:{h:.2f}px"></div>'
            f'<div class="contrib-label"><span>{esc_name}</span></div></div>'
        )

    lines.append("  </div>")
    lines.append("</div>")
    lines.append("```")
    lines.append("")
    lines.append("::: {.contrib-note}")
    lines.append("Tip: scroll horizontally and hover each bar to see exact paper counts.")
    lines.append(":::")
    return "\n".join(lines)


def build_details(papers: List[Dict]) -> str:
    lines: List[str] = []
    min_year = 2006

    def resolved_year(p: Dict) -> Optional[int]:
        pub_date = p.get("publication_date")
        y = p.get("year")
        if pub_date and str(pub_date)[:4].isdigit():
            return int(str(pub_date)[:4])
        if isinstance(y, int):
            return y
        return None

    def year_bucket(p: Dict) -> str:
        y = resolved_year(p)
        if y is not None:
            return str(y)
        return "Unknown"

    papers_for_details: List[Dict] = []
    for p in papers:
        y = resolved_year(p)
        if y is not None and y < min_year:
            continue
        papers_for_details.append(p)

    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for p in papers_for_details:
        grouped[year_bucket(p)].append(p)

    def sort_year_key(y: str) -> Tuple[int, int, str]:
        if y.isdigit():
            return (0, -int(y), "")
        if y.lower() == "unknown":
            return (2, 0, "")
        return (1, 0, y.lower())

    ordered_years = sorted(grouped.keys(), key=sort_year_key)
    lines.append("## Papers")
    lines.append("")
    lines.append(
        f"Papers are grouped by publication year. Expand a year to browse papers. "
        f"Only papers from **{min_year}+** are shown (plus records with unknown year)."
    )
    lines.append("")
    lines.append("```{=html}")
    lines.append("<style>")
    lines.append(".paper-year { margin: 0 0 10px 0; border: 1px solid #e5e7eb; border-radius: 8px; padding: 6px 10px; background: #fff; }")
    lines.append(".paper-year > summary { cursor: pointer; font-size: 1.0rem; }")
    lines.append(".paper-year > .paper-item { margin: 8px 0; }")
    lines.append(".paper-year > .paper-item > summary { cursor: pointer; }")
    lines.append("</style>")

    for y in ordered_years:
        papers_in_year = grouped[y]
        year_label = html.escape(y)
        year_count = len(papers_in_year)
        year_summary = f"{year_label} ({year_count} paper{'s' if year_count != 1 else ''})"
        lines.append("<details class=\"paper-year\">")
        lines.append(f"  <summary><strong>{year_summary}</strong></summary>")
        lines.append("")

        for p in papers_in_year:
            title = normalize_space(p.get("title", "Untitled"))
            authors = p.get("authors") or []
            authors_str = ", ".join(authors) if authors else "Unknown"
            year = p.get("year")
            pub_date = p.get("publication_date")
            url = p.get("url") or ""
            abstract = normalize_space(p.get("abstract") or "")
            if not abstract:
                abstract = "Abstract unavailable from Semantic Scholar."
            keywords = [normalize_space(k) for k in (p.get("matched_keywords") or []) if normalize_space(k)]
            keywords_str = ", ".join(keywords) if keywords else "None"
            venue = normalize_space(p.get("venue") or "")
            doi = p.get("doi")

            date_label = pub_date if pub_date else (str(year) if year else "Unknown")
            summary = html.escape(f"{date_label} — {title}")

            lines.append('<details class="paper-item">')
            lines.append(f"  <summary><strong>{summary}</strong></summary>")
            lines.append("")
            lines.append("<p>")
            lines.append(f"<strong>Authors</strong>: {html.escape(authors_str)}<br>")
            if year is not None:
                lines.append(f"<strong>Year</strong>: {html.escape(str(year))}<br>")
            if pub_date:
                lines.append(f"<strong>Publication Date</strong>: {html.escape(str(pub_date))}<br>")
            if venue:
                lines.append(f"<strong>Venue</strong>: {html.escape(venue)}<br>")
            if doi:
                lines.append(f"<strong>DOI</strong>: {html.escape(str(doi))}<br>")
            if url:
                lines.append(
                    f'<strong>Link</strong>: <a href="{html.escape(url, quote=True)}">Semantic Scholar</a><br>'
                )
            lines.append(f"<strong>Matched Keywords</strong>: {html.escape(keywords_str)}")
            lines.append("</p>")
            lines.append("<p><strong>Abstract</strong>:<br>")
            lines.append(f"{html.escape(abstract)}")
            lines.append("</p>")
            lines.append("</details>")
            lines.append("")

        lines.append("</details>")
        lines.append("")

    lines.append("```")

    return "\n".join(lines).rstrip() + "\n"


def sort_papers(papers: List[Dict]) -> List[Dict]:
    def key(p: Dict):
        pd = parse_date(p.get("publication_date"))
        y = p.get("year")
        if pd:
            return (pd.toordinal(), normalize_space(p.get("title", "")).lower())
        if isinstance(y, int):
            return (date(y, 1, 1).toordinal(), normalize_space(p.get("title", "")).lower())
        return (0, normalize_space(p.get("title", "")).lower())

    return sorted(papers, key=key, reverse=True)


def generate_qmd(data_path: Path, blocked_path: Path) -> str:
    obj = json.loads(data_path.read_text(encoding="utf-8"))
    blocked = load_blocklist(blocked_path)
    papers, _ = filter_blocked_papers(obj.get("papers") or [], blocked)
    papers = sort_papers(papers)
    generated_at = obj.get("generated_at") or datetime.utcnow().isoformat() + "Z"
    paper_count = len(papers)
    keywords = obj.get("keywords") or []

    parts: List[str] = []
    parts.append("---")
    parts.append('title: "TMLE Papers"')
    parts.append("page-layout: full")
    parts.append("---")
    parts.append("")
    parts.append("Auto-updated list of TMLE-related papers tracked from Semantic Scholar queries.")
    parts.append("")
    parts.append(
        "_Note: this list is not complete and may include irrelevant articles; it will be refined over time._"
    )
    parts.append("")
    parts.append(f"Last refreshed: {generated_at}")
    parts.append("")
    parts.append(f"Keywords queried: {', '.join(keywords)}")
    parts.append("")
    parts.append(f"Total tracked papers: **{paper_count}**")
    parts.append("")
    parts.append("<!-- monthly-chart:start -->")
    parts.append("")
    parts.append(build_monthly_chart(papers))
    parts.append("")
    parts.append("<!-- monthly-chart:end -->")
    parts.append("")
    parts.append("<!-- contributor-chart:start -->")
    parts.append("")
    parts.append(build_contributor_chart(papers))
    parts.append("")
    parts.append("<!-- contributor-chart:end -->")
    parts.append("")
    parts.append(build_details(papers))
    return "\n".join(parts)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate tmle-papers.qmd from Semantic Scholar JSON.")
    p.add_argument(
        "--input",
        default="data/semantic_scholar_tmle_papers.json",
        help="Input JSON path.",
    )
    p.add_argument(
        "--output",
        default="tmle-papers.qmd",
        help="Output qmd path.",
    )
    p.add_argument(
        "--blocked-json",
        default=DEFAULT_BLOCKLIST_JSON,
        help="JSON file listing papers to exclude (by paper_id / doi / title).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    blocked_path = Path(args.blocked_json)
    if not input_path.exists():
        raise SystemExit(f"Input JSON not found: {input_path}")
    out = generate_qmd(input_path, blocked_path)
    output_path.write_text(out, encoding="utf-8")
    print(f"Wrote {output_path} from {input_path} (blocklist: {blocked_path})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
