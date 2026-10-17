#!/usr/bin/env python3
"""Build the review PR body and complete new-paper list without network access.

The final catalog, compared with the reviewed default-branch baseline, is the
source of truth. Run-report counts can describe intermediate discovery records;
they must not decide which papers are new in the proposed PR.
"""
import argparse
import html
import json
import re
from pathlib import Path
from urllib.parse import quote, urlsplit

from fetch_semantic_scholar_papers import _paper_doi_from_record, _record_aliases

# A conservative UTF-8 byte budget, below GitHub's 65,536-character body limit.
# Leave room for workflow notices appended after generation.
PR_BODY_BUDGET = 60_000


def plain_text(value):
    return " ".join(value.split()) if isinstance(value, str) else ""


def markdown_text(value):
    """Keep metadata as one literal line, never Markdown or HTML instructions."""
    value = html.escape(plain_text(value), quote=False)
    return re.sub(r"([\\`*_{}\[\]()#+\-.!|@~$])", r"\\\1", value)


def catalog_papers(catalog):
    if not isinstance(catalog, dict) or not isinstance(catalog.get("papers"), list):
        raise ValueError("Catalog must contain a papers list.")
    if any(not isinstance(paper, dict) for paper in catalog["papers"]):
        raise ValueError("Catalog papers must be objects.")
    return catalog["papers"]


def newly_added_papers(baseline, current):
    """Use the fetcher's identity aliases so metadata enrichment is not new."""
    old_aliases = {
        alias for paper in catalog_papers(baseline) for alias in _record_aliases(paper)
    }
    result = []
    for paper in catalog_papers(current):
        aliases = _record_aliases(paper)
        if not aliases:
            raise ValueError("A proposed paper has no usable identity.")
        if old_aliases.isdisjoint(aliases):
            result.append(paper)
    return result


def safe_http_url(value):
    if not isinstance(value, str) or any(ord(c) < 32 for c in value):
        return None
    try:
        parsed = urlsplit(value.strip())
        if (parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname
                or parsed.username is not None or parsed.password is not None):
            return None
    except ValueError:
        return None
    # Encode spaces, angle brackets and parentheses before use in Markdown.
    return quote(value.strip(), safe=":/?#[]@!$&'*+,;=%")


def paper_url(paper):
    doi = _paper_doi_from_record(paper)
    if doi and re.fullmatch(r"10\.\d{4,9}/\S+", doi):
        return "https://doi.org/" + quote(doi, safe="/")
    url = safe_http_url(paper.get("url"))
    if url:
        return url
    paper_id = paper.get("paper_id") or paper.get("paperId") or paper.get("id")
    if isinstance(paper_id, str) and paper_id.strip():
        return "https://www.semanticscholar.org/paper/" + quote(paper_id.strip(), safe="")
    return None


def authors_text(paper):
    authors = paper.get("authors")
    if not isinstance(authors, list) or not authors:
        return "Author names unavailable in source metadata."
    names = []
    for author in authors:
        name = author.get("name") if isinstance(author, dict) else author
        names.append(markdown_text(name) or "[Author name unavailable in source metadata]")
    # Do not limit the author count, abbreviate to et al., or silently drop gaps.
    return "; ".join(names)


def render_paper_list(papers):
    lines = [f"## Newly added papers ({len(papers)})", "",
             "Compared with the default-branch catalog at the start of this run. "
             "All author names supplied by the source are shown.", ""]
    if not papers:
        lines.append("No newly added papers; this proposal only updates existing records or catalog content.")
    for number, paper in enumerate(papers, 1):
        title = markdown_text(paper.get("title")) or "Title unavailable in source metadata"
        url = paper_url(paper)
        heading = f"[{title}](<{url}>)" if url else title
        lines += [f"{number}. {heading}", f"   - Authors: {authors_text(paper)}"]
        if not url:
            lines.append("   - Paper link unavailable in source metadata.")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_summary(report, baseline, current, run_url):
    run_url = safe_http_url(run_url)
    if not run_url:
        raise ValueError("A valid HTTP(S) workflow run URL is required.")
    artifact_link = f"[workflow run artifacts](<{run_url}#artifacts>)"
    lines = ["## TMLE discovery review", ""]
    details = ""
    complete_list = None
    if report is None:
        lines.append("No discovery report was produced. Check the failed step; no review PR is created.")
    else:
        if not isinstance(report, dict):
            raise ValueError("Discovery report must be an object.")
        for label, key in [
            ("Catalog papers", "paper_count"),
            ("Newly discovered papers (fetch report)", "newly_appended_count"),
            ("Updated existing papers", "updated_existing_count"),
            ("Blocked existing papers removed", "existing_blocked_removed"),
            ("Existing duplicates removed", "existing_duplicates_removed"),
        ]:
            value = report.get(key)
            lines.append(f'- {label}: {value if type(value) is int else "not available"}')
        lines.append(f'- Incomplete: {report.get("incomplete", True) is not False}')
        lines.append(f'- Query errors: {len(report.get("query_errors", []))}')
        lines.append(f'- Truncated queries: {len(report.get("query_truncations", []))}')
        diagnostics = {key: report.get(key) for key in
                       ("query_stats", "query_errors", "query_truncations", "incomplete")}
        details = ("\n<details><summary>Query diagnostics</summary>\n\n<pre>"
                   + html.escape(json.dumps(diagnostics, ensure_ascii=False, indent=2))
                   + "</pre>\n</details>\n")
        if (report.get("incomplete") is False and report.get("output_updated") is True
                and not report.get("query_errors") and not report.get("query_truncations")):
            if not isinstance(current, dict) or current.get("incomplete") is not False:
                raise ValueError("A complete final catalog is required for the new-paper list.")
            papers = newly_added_papers(baseline, current)
            complete_list = render_paper_list(papers)
            lines.append(f"- New papers in this PR (compared with baseline): {len(papers)}")
            paper_section = "\n" + complete_list
        else:
            paper_section = ("\nThe new-paper list is unavailable because discovery did not "
                             "complete successfully; no review PR is created.\n")
    if report is None:
        paper_section = ""
    footer = (f"\n[Workflow run and diagnostic artifacts](<{run_url}>)\n\n"
              "The `tmle-discovery-diagnostics` artifact includes the complete run report, "
              "query diagnostics, and `newly-added-papers.md` after successful discovery. "
              "Artifacts are retained for 30 days.\n\n"
              "Review relevance, dates, duplicates, blocked records, and the rendered page before merging.\n"
              "This workflow does not merge or publish the changes.\n")
    prefix = "\n".join(lines) + "\n"
    summary = prefix + paper_section + details + footer
    if len(summary.encode("utf-8")) > PR_BODY_BUDGET:
        # Prefer the full new-paper list over verbose query diagnostics.
        details = f"\nComplete query diagnostics are in `report.json` in the {artifact_link}.\n"
        summary = prefix + paper_section + details + footer
    if len(summary.encode("utf-8")) > PR_BODY_BUDGET:
        if complete_list is None:
            raise ValueError("Summary exceeds the PR body budget without a complete paper list.")
        paper_section = (
            f"\n## Newly added papers ({len(papers)})\n\n"
            "The complete list exceeds the GitHub PR-description size budget. "
            "No papers or authors have been dropped from the saved list. "
            f"Download **tmle-discovery-diagnostics** from the {artifact_link} "
            "and open **newly-added-papers.md** for every title, all available author names, "
            "and paper links. Artifacts are retained for 30 days.\n"
        )
        summary = prefix + paper_section + details + footer
    if len(summary.encode("utf-8")) > PR_BODY_BUDGET:
        raise ValueError("Summary still exceeds the PR body budget.")
    return summary, complete_list


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-url", required=True)
    parser.add_argument("--step-summary", type=Path)
    args = parser.parse_args(argv)
    report = json.loads(args.report.read_text(encoding="utf-8")) if args.report.exists() else None
    successful = (isinstance(report, dict) and report.get("incomplete") is False
                  and report.get("output_updated") is True
                  and not report.get("query_errors") and not report.get("query_truncations"))
    baseline = json.loads(args.baseline.read_text(encoding="utf-8")) if successful else None
    current = json.loads(args.catalog.read_text(encoding="utf-8")) if successful else None
    summary, paper_list = build_summary(report, baseline, current, args.run_url)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "review-summary.md").write_text(summary, encoding="utf-8")
    if paper_list is not None:
        (args.output_dir / "newly-added-papers.md").write_text(paper_list, encoding="utf-8")
    if args.step_summary:
        with args.step_summary.open("a", encoding="utf-8") as output:
            output.write(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
