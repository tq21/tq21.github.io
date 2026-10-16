"""Offline tests for complete, safe weekly paper-review descriptions."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import generate_tmle_review_summary as review

RUN_URL = "https://github.com/tq21/tq21.github.io/actions/runs/123"
REPORT = {"incomplete": False, "output_updated": True, "paper_count": 2,
          "newly_appended_count": 1, "updated_existing_count": 1,
          "query_errors": [], "query_truncations": [], "query_stats": []}


def paper(number, **overrides):
    value = {"paper_id": str(number), "title": f"Paper {number}", "year": 2026,
             "authors": ["First Author", "Second Author", "Third Author"],
             "doi": f"10.1234/paper-{number}"}
    value.update(overrides)
    return value


def catalog(*papers):
    return {"incomplete": False, "papers": list(papers)}


class PaperDifferenceTests(unittest.TestCase):
    def test_final_catalog_not_report_controls_full_list(self):
        before = paper(1)
        updated = paper(1, authors=["Updated Author"], title="Updated title")
        added = [paper(n) for n in range(2, 19)]
        # Deliberately stale/incomplete intermediate report metadata must not win.
        report = {**REPORT, "newly_appended_count": 999,
                  "newly_added_papers": [paper(999)]}
        summary, full = review.build_summary(report, catalog(before),
                                            catalog(updated, *added), RUN_URL)
        self.assertIn("New papers in this PR (compared with baseline): 17", summary)
        self.assertNotIn("Updated Author", summary)
        self.assertNotIn("Paper 999", summary)
        for number, added_paper in enumerate(added, 1):
            self.assertIn(f"{number}. [Paper {added_paper['paper_id']}]", summary)
            for author in added_paper["authors"]:
                self.assertIn(author, summary)
        self.assertIn(full, summary)

    def test_metadata_enrichment_matches_each_fetcher_identity_alias(self):
        before = paper(1)
        candidates = [
            paper(1, title="Renamed", doi=None),  # provider ID
            paper(2, title="Renamed", doi="https://doi.org/10.1234/PAPER-1"),
            paper(3, title="PAPER 1!", doi=None),  # normalized title
            paper(4, title="Paper 1", doi=None, year=2025),
        ]
        for candidate in candidates:
            with self.subTest(candidate=candidate):
                self.assertEqual(review.newly_added_papers(catalog(before), catalog(candidate)), [])

    def test_deletions_and_record_updates_are_not_new(self):
        summary, full = review.build_summary(REPORT, catalog(paper(1), paper(2)),
                                            catalog(paper(1, authors=["Changed"])), RUN_URL)
        self.assertIn("Newly added papers (0)", summary)
        self.assertIn("No newly added papers", full)
        self.assertNotIn("Changed", summary)

    def test_bad_catalog_fails_instead_of_publishing_an_incomplete_list(self):
        for bad in [{}, {"papers": "no"}, {"papers": [None]}, {"papers": [{}]}]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                review.newly_added_papers(catalog(), bad)


class PaperRenderingTests(unittest.TestCase):
    def test_every_author_is_in_order_without_et_al_or_count_cap(self):
        authors = [f"Researcher {number}" for number in range(200)]
        result = review.render_paper_list([paper(1, authors=authors)])
        self.assertIn("; ".join(authors), result)
        self.assertNotIn("et al.", result)
        self.assertIn("https://doi.org/10.1234/paper-1", result)

    def test_missing_authors_or_names_are_flagged_not_invented_or_omitted(self):
        for value in [None, [], "not a list"]:
            with self.subTest(value=value):
                self.assertEqual(review.authors_text({"authors": value}),
                                 "Author names unavailable in source metadata.")
        result = review.authors_text({"authors": [{"name": "Ada"}, {}, None, "", "Grace"]})
        self.assertTrue(result.startswith("Ada; "))
        self.assertTrue(result.endswith("; Grace"))
        self.assertEqual(result.count("Author name unavailable"), 3)

    def test_markdown_html_newlines_and_mentions_are_literal_metadata(self):
        result = review.render_paper_list([paper(
            1, title="[Injected](https://example.com)\n## <script>x</script> & *bold*",
            authors=["@everyone", "A_[B]", {"name": "C\n- D"}, "Back\\slash", "~~strike~~ $math$"],
        )])
        self.assertIn(r"\[Injected\]\(https://example\.com\) \#\# &lt;script&gt;x&lt;/script&gt; &amp; \*bold\*", result)
        self.assertIn(r"\@everyone; A\_\[B\]; C \- D; Back\\slash", result)
        self.assertIn(r"\~\~strike\~\~ \$math\$", result)
        self.assertNotIn("<script>", result)
        self.assertNotIn("\n## <", result)

    def test_unicode_title_and_author_names_are_preserved(self):
        result = review.render_paper_list([paper(1, title="统计学与TMLE", authors=["邱天", "Élodie Łukasz"])])
        self.assertIn("统计学与TMLE", result)
        self.assertIn("邱天; Élodie Łukasz", result)

    def test_links_prioritize_doi_then_source_url_then_provider_id(self):
        self.assertEqual(review.paper_url(paper(1, doi="https://doi.org/10.1234/A(B)")),
                         "https://doi.org/10.1234/a%28b%29")
        self.assertEqual(review.paper_url(paper(1, doi=None, url="https://example.org/a (b)")),
                         "https://example.org/a%20%28b%29")
        for url in ["javascript:alert(1)", "data:text/html,hello", "https://a\n.example.com",
                    "https://user:password@example.com", "https://[", "//example.com"]:
            with self.subTest(url=url):
                self.assertEqual(review.paper_url(paper("a/b", doi=None, url=url)),
                                 "https://www.semanticscholar.org/paper/a%2Fb")
        self.assertIsNone(review.paper_url({}))
        result = review.render_paper_list([{"title": "No link", "authors": []}])
        self.assertIn("Paper link unavailable in source metadata.", result)


class SummaryAndCLITests(unittest.TestCase):
    def test_failed_or_unwritten_discovery_never_claims_a_new_paper_list(self):
        for report in [None, {**REPORT, "incomplete": True},
                       {**REPORT, "output_updated": False},
                       {**REPORT, "query_errors": ["error"]},
                       {**REPORT, "query_truncations": ["truncated"]}]:
            with self.subTest(report=report):
                summary, full = review.build_summary(report, None, None, RUN_URL)
                self.assertIsNone(full)
                self.assertIn("no review pr is created", summary.lower())
                self.assertNotIn("## Newly added papers", summary)

    def test_complete_report_requires_complete_catalog(self):
        with self.assertRaises(ValueError):
            review.build_summary(REPORT, catalog(), {"papers": [paper(1)]}, RUN_URL)

    def test_large_diagnostics_are_moved_out_before_any_paper_list(self):
        report = {**REPORT, "query_stats": [{"huge": "x" * 100_000}]}
        summary, full = review.build_summary(report, catalog(), catalog(paper(1)), RUN_URL)
        self.assertLessEqual(len(summary.encode("utf-8")), review.PR_BODY_BUDGET)
        self.assertIn(full, summary)
        self.assertIn("Complete query diagnostics are in", summary)
        self.assertNotIn("complete list exceeds", summary)

    def test_huge_lists_use_explicit_full_artifact_link_without_truncating_saved_list(self):
        papers = [paper(n, title=f"Title {n} " + "统计学" * 60,
                        authors=[f"Author {n}-{author}" for author in range(30)])
                  for n in range(700)]
        summary, full = review.build_summary(REPORT, catalog(), catalog(*papers), RUN_URL)
        self.assertLessEqual(len(summary.encode("utf-8")), review.PR_BODY_BUDGET)
        self.assertIn("Newly added papers (700)", summary)
        self.assertIn("complete list exceeds", summary)
        self.assertIn(RUN_URL + "#artifacts", summary)
        self.assertIn("**newly-added-papers.md**", summary)
        self.assertIn("retained for 30 days", summary)
        for p in papers:
            self.assertIn(review.markdown_text(p["title"]), full)
            for name in p["authors"]:
                self.assertIn(review.markdown_text(name), full)
        self.assertIn("700. [Title 699", full)

    def test_normal_cli_writes_pr_body_full_list_and_workflow_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, value in [("baseline", catalog(paper(1))),
                                ("current", catalog(paper(1), paper(2))), ("report", REPORT)]:
                (root / f"{name}.json").write_text(json.dumps(value), encoding="utf-8")
            args = ["--baseline", str(root / "baseline.json"), "--catalog", str(root / "current.json"),
                    "--report", str(root / "report.json"), "--output-dir", str(root / "out"),
                    "--run-url", RUN_URL, "--step-summary", str(root / "step.md")]
            self.assertEqual(review.main(args), 0)
            summary = (root / "out/review-summary.md").read_text(encoding="utf-8")
            full = (root / "out/newly-added-papers.md").read_text(encoding="utf-8")
            self.assertIn(full, summary)
            self.assertEqual(summary, (root / "step.md").read_text(encoding="utf-8"))
            self.assertIn("[Paper 2]", summary)
            self.assertNotIn("[Paper 1]", summary)

    def test_missing_report_cli_still_emits_diagnostics_without_reading_catalogs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = ["--baseline", str(root / "missing"), "--catalog", str(root / "missing"),
                    "--report", str(root / "missing"), "--output-dir", str(root / "out"),
                    "--run-url", RUN_URL]
            self.assertEqual(review.main(args), 0)
            self.assertTrue((root / "out/review-summary.md").is_file())
            self.assertFalse((root / "out/newly-added-papers.md").exists())

    def test_query_diagnostics_are_html_escaped(self):
        summary, _ = review.build_summary({**REPORT, "query_stats": [{"keyword": "</pre><script>"}]},
                                         catalog(), catalog(), RUN_URL)
        self.assertNotIn("</pre><script>", summary)
        self.assertIn("&lt;/pre&gt;&lt;script&gt;", summary)


if __name__ == "__main__":
    unittest.main()
