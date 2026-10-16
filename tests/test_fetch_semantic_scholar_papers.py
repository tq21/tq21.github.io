"""Offline regression tests for safe, complete TMLE discovery."""
import io
import json
import sys
import tempfile
import unittest
import urllib.error
import urllib.parse
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import fetch_semantic_scholar_papers as fetch
import generate_tmle_papers_qmd as generate


def raw_paper(number, **overrides):
    value = {
        "paperId": str(number), "title": f"TMLE paper {number}", "abstract": "TMLE methods",
        "year": 2026, "publicationDate": "2026-10-01", "authors": [{"name": "A. Researcher"}],
        "externalIds": {"DOI": f"10.1234/{number}"},
    }
    value.update(overrides)
    return value


OPTIONS = dict(keyword="super learning", fields=fetch.DEFAULT_FIELDS, page_size=1000,
               max_results_per_keyword=0, request_interval_s=0, headers={}, ssl_context=None,
               timeout_s=1, max_retries=0, backoff_s=1, max_backoff_s=1)


class BulkPaginationTests(unittest.TestCase):
    def run_fetch(self, responses, **overrides):
        with patch.object(fetch, "_http_get_json", side_effect=responses) as http:
            result = fetch.fetch_keyword_bulk(**{**OPTIONS, **overrides})
        return result, http

    def test_exhausts_short_pages_newest_first_without_unsupported_limit(self):
        result, http = self.run_fetch([
            {"total": 3, "data": [raw_paper(1)], "token": "second"},
            {"total": 3, "data": [raw_paper(2)], "token": "third"},
            {"total": 3, "data": [raw_paper(3, publicationDate=None)]},
        ])
        self.assertEqual(len(result.papers), 3)
        self.assertEqual(result.pages_fetched, 3)
        self.assertFalse(result.truncated)
        for call in http.call_args_list:
            params = urllib.parse.parse_qs(urllib.parse.urlparse(call.kwargs["url"]).query)
            self.assertEqual(params["sort"], ["publicationDate:desc"])
            self.assertEqual(params["query"], ["super learning"])
            self.assertNotIn("limit", params)
            self.assertNotIn("year", params)
        self.assertIn("token=second", http.call_args_list[1].kwargs["url"])
        self.assertIn("token=third", http.call_args_list[2].kwargs["url"])

    def test_default_unlimited_fetch_exceeds_old_1500_cap(self):
        result, _ = self.run_fetch([
            {"data": [raw_paper(i) for i in range(1000)], "token": "next"},
            {"data": [raw_paper(i) for i in range(1000, 2001)]},
        ])
        self.assertEqual(len(result.papers), 2001)
        self.assertFalse(result.truncated)

    def test_cap_exactly_enforced_even_when_api_ignores_page_size(self):
        result, _ = self.run_fetch([
            {"data": [raw_paper(i) for i in range(1000)], "token": "next"},
            {"data": [raw_paper(i) for i in range(1000, 2000)], "token": "last"},
        ], max_results_per_keyword=1500)
        self.assertEqual(len(result.papers), 1500)
        self.assertTrue(result.truncated)
        self.assertIn("cap", result.truncation_reason)

    def test_cap_at_page_boundary_with_token_is_incomplete(self):
        result, _ = self.run_fetch([{"data": [raw_paper(1)], "token": "next"}],
                                   max_results_per_keyword=1)
        self.assertTrue(result.truncated)

    def test_cap_at_exhaustion_is_complete(self):
        result, _ = self.run_fetch([{"data": [raw_paper(1)]}], max_results_per_keyword=1)
        self.assertFalse(result.truncated)

    def test_oversized_final_page_is_incomplete_even_without_token(self):
        result, _ = self.run_fetch([{"data": [raw_paper(1), raw_paper(2)]}],
                                   max_results_per_keyword=1)
        self.assertTrue(result.truncated)
        self.assertEqual(len(result.papers), 1)

    def test_estimated_total_does_not_replace_continuation(self):
        result, _ = self.run_fetch([
            {"total": 1, "data": [raw_paper(1)], "token": "next"},
            {"total": 1, "data": [raw_paper(2)]},
        ])
        self.assertEqual(len(result.papers), 2)
        self.assertFalse(result.truncated)

    def test_empty_exhausted_query_is_valid(self):
        result, _ = self.run_fetch([{"total": 0, "data": []}])
        self.assertEqual(result.papers, [])
        self.assertFalse(result.truncated)

    def test_empty_page_with_continuation_fails(self):
        with self.assertRaisesRegex(RuntimeError, "empty page"):
            self.run_fetch([{"data": [], "token": "next"}])

    def test_repeated_token_fails_instead_of_looping(self):
        with self.assertRaisesRegex(RuntimeError, "repeated"):
            self.run_fetch([
                {"data": [raw_paper(1)], "token": "same"},
                {"data": [raw_paper(2)], "token": "same"},
            ])

    def test_cyclic_tokens_fail(self):
        with self.assertRaisesRegex(RuntimeError, "repeated"):
            self.run_fetch([
                {"data": [raw_paper(1)], "token": "a"},
                {"data": [raw_paper(2)], "token": "b"},
                {"data": [raw_paper(3)], "token": "a"},
            ])

    def test_malformed_responses_fail(self):
        for response in [{"error": "failure"}, {"data": None}, {"data": ["not an object"]},
                         {"data": [], "token": 123}, {"data": [{}]},
                         {"data": [raw_paper(1, title=["TMLE"])]},
                         {"data": [raw_paper(1, authors=["A. Author"])]},
                         {"data": [raw_paper(1, externalIds=[])]},
                         {"data": [raw_paper(1, year=[])]}, []]:
            with self.subTest(response=response), self.assertRaises(RuntimeError):
                self.run_fetch([response])


class RelevancePaginationTests(unittest.TestCase):
    def test_uses_next_even_for_short_pages(self):
        with patch.object(fetch, "_http_get_json", side_effect=[
            {"total": 2, "data": [raw_paper(1)], "next": 1},
            {"total": 2, "data": [raw_paper(2)]},
        ]) as http:
            result = fetch.fetch_keyword(**{**OPTIONS, "page_size": 100})
        self.assertEqual(len(result.papers), 2)
        self.assertFalse(result.truncated)
        self.assertIn("offset=1", http.call_args_list[1].kwargs["url"])

    def test_service_limit_is_reported_as_truncation(self):
        responses = [{"total": 1001, "data": [raw_paper(i) for i in range(n, n + 100)],
                      "next": n + 100} for n in range(0, 1000, 100)]
        with patch.object(fetch, "_http_get_json", side_effect=responses):
            result = fetch.fetch_keyword(**{**OPTIONS, "page_size": 100})
        self.assertEqual(len(result.papers), 1000)
        self.assertTrue(result.truncated)

    def test_forward_jump_cannot_masquerade_as_completion(self):
        with patch.object(fetch, "_http_get_json", return_value={
            "total": 1001, "data": [raw_paper(1)], "next": 1000,
        }), self.assertRaisesRegex(RuntimeError, "continuation offset"):
            fetch.fetch_keyword(**{**OPTIONS, "page_size": 100})

    def test_bad_offset_fails(self):
        with patch.object(fetch, "_http_get_json", return_value={
            "total": 10, "data": [raw_paper(1)], "next": 0,
        }), self.assertRaisesRegex(RuntimeError, "continuation offset"):
            fetch.fetch_keyword(**{**OPTIONS, "page_size": 100})


class MainSafetyTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.output = self.root / "papers.json"
        self.report = self.root / "report.json"
        self.blocked = self.root / "blocked.json"
        self.blocked.write_text('{"titles": ["Blocked TMLE paper"]}')
        records, _ = fetch.dedupe_and_filter([("tmle", raw_paper(1))], min_year=2006,
                                            blocked=fetch.load_blocklist(self.blocked))
        self.baseline = {"generated_at": "2026-10-09T00:00:00Z", "papers": [fetch.asdict(records[0])]}
        self.output.write_text(json.dumps(self.baseline))
        self.original = self.output.read_bytes()

    def invoke(self, responses, extra=()):
        argv = ["fetch", "--output", str(self.output), "--report", str(self.report),
                "--blocked-json", str(self.blocked), "--keywords", "tmle", "super learning",
                "--request-interval-s", "0", *extra]
        with patch.object(sys, "argv", argv), patch.object(fetch, "fetch_keyword_bulk", side_effect=responses):
            return fetch.main()

    def test_success_writes_output_and_report_without_full_records(self):
        code = self.invoke([fetch.QueryResult([raw_paper(1)], 1, 1),
                            fetch.QueryResult([raw_paper(2)], 1, 1)])
        result = json.loads(self.output.read_text())
        report = json.loads(self.report.read_text())
        self.assertEqual(code, 0)
        self.assertEqual(result["paper_count"], 2)
        self.assertEqual(result["newly_appended_count"], 1)
        self.assertFalse(report["incomplete"])
        self.assertTrue(report["output_updated"])
        self.assertEqual(len(report["query_stats"]), 2)
        self.assertNotIn("papers", report)
        self.assertEqual(result["sort"], "publicationDate:desc")

    def test_truncation_does_not_overwrite_existing_output(self):
        code = self.invoke([fetch.QueryResult([raw_paper(2)], 1, 2, True, "cap"),
                            fetch.QueryResult([], 1, 0)])
        self.assertEqual(code, 2)
        self.assertEqual(self.output.read_bytes(), self.original)
        report = json.loads(self.report.read_text())
        self.assertTrue(report["incomplete"])
        self.assertFalse(report["output_updated"])
        self.assertEqual(report["query_truncations"], [{"keyword": "tmle", "reason": "cap"}])

    def test_failed_query_does_not_overwrite_existing_output(self):
        code = self.invoke([RuntimeError("HTTP 429")])
        self.assertEqual(code, 2)
        self.assertEqual(self.output.read_bytes(), self.original)
        self.assertEqual(len(json.loads(self.report.read_text())["query_errors"]), 1)

    def test_continue_on_error_still_fails_safely(self):
        code = self.invoke([RuntimeError("HTTP 429"), fetch.QueryResult([raw_paper(2)], 1, 1)],
                           ["--continue-on-error"])
        self.assertEqual(code, 2)
        self.assertEqual(self.output.read_bytes(), self.original)
        self.assertEqual(len(json.loads(self.report.read_text())["query_stats"]), 1)

    def test_existing_and_fetched_blocked_records_excluded(self):
        code = self.invoke([fetch.QueryResult([raw_paper(2, title="Blocked TMLE paper")], 1, 1),
                            fetch.QueryResult([], 1, 0)])
        self.assertEqual(code, 0)
        result = json.loads(self.output.read_text())
        self.assertEqual(result["paper_count"], 1)
        self.assertEqual(result["blocked_match_count"], 1)

    def test_unwritable_report_preserves_catalog(self):
        self.report.mkdir()
        code = self.invoke([fetch.QueryResult([raw_paper(2)], 1, 1), fetch.QueryResult([], 1, 0)])
        self.assertEqual(code, 2)
        self.assertEqual(self.output.read_bytes(), self.original)

    def test_malformed_api_paper_fails_with_diagnostic_report(self):
        argv = ["fetch", "--output", str(self.output), "--report", str(self.report),
                "--blocked-json", str(self.blocked), "--keywords", "tmle"]
        with patch.object(sys, "argv", argv), patch.object(fetch, "_http_get_json", return_value={"data": [{}]}):
            code = fetch.main()
        self.assertEqual(code, 2)
        self.assertEqual(self.output.read_bytes(), self.original)
        self.assertTrue(json.loads(self.report.read_text())["incomplete"])

    def test_default_is_exhaustive(self):
        with patch.object(sys, "argv", ["fetch"]):
            self.assertEqual(fetch.parse_args().max_results_per_keyword, 0)

    def test_invalid_options_rejected(self):
        for extra in [["--max-results-per-keyword", "-1"], ["--keywords"],
                      ["--request-interval-s", "-1"], ["--max-retries", "-1"],
                      ["--timeout-s", "0"], ["--report", str(self.output)]]:
            with self.subTest(extra=extra), self.assertRaises(SystemExit):
                self.invoke([], extra)


class RetryAndOutputTests(unittest.TestCase):
    def request(self):
        return fetch._http_get_json(url="https://example.test", headers={}, ssl_context=None,
                                    timeout_s=1, max_retries=1, backoff_s=2, max_backoff_s=10)

    def response(self):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = b'{"data": []}'
        return response

    def test_429_retry_after_is_honored(self):
        error = urllib.error.HTTPError("https://example.test", 429, "Limited",
                                       {"Retry-After": "4"}, io.BytesIO())
        with patch.object(fetch.urllib.request, "urlopen", side_effect=[error, self.response()]), \
             patch.object(fetch.time, "sleep") as sleep:
            self.assertEqual(self.request(), {"data": []})
        sleep.assert_called_once_with(4.0)

    def test_network_timeout_is_retried(self):
        with patch.object(fetch.urllib.request, "urlopen", side_effect=[TimeoutError(), self.response()]), \
             patch.object(fetch.time, "sleep") as sleep:
            self.assertEqual(self.request(), {"data": []})
        sleep.assert_called_once_with(2)

    def test_permanent_http_error_is_not_retried(self):
        error = urllib.error.HTTPError("https://example.test", 403, "Forbidden", {}, io.BytesIO(b"denied"))
        with patch.object(fetch.urllib.request, "urlopen", side_effect=error) as http, \
             self.assertRaisesRegex(RuntimeError, "HTTP 403"):
            self.request()
        self.assertEqual(http.call_count, 1)

    def test_atomic_replace_failure_preserves_file_and_cleans_temp(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "papers.json"
            path.write_text("old")
            with patch.object(Path, "replace", side_effect=OSError("read only")), \
                 self.assertRaises(OSError):
                fetch.write_json_atomic(path, {"new": True})
            self.assertEqual(path.read_text(), "old")
            self.assertEqual(list(Path(directory).iterdir()), [path])


class ExistingBehaviorTests(unittest.TestCase):
    def test_append_replay_is_idempotent_and_aliases_deduplicate(self):
        records, _ = fetch.dedupe_and_filter([("tmle", raw_paper(1))], min_year=2006,
                                            blocked={"paper_ids": set(), "dois": set(), "titles": set()})
        result = fetch.append_new_records_only(existing_papers=[fetch.asdict(records[0])],
                                               fetched_records=records)
        self.assertEqual((len(result[0]), *result[1:]), (1, 0, 0, 0, []))

    def test_super_learning_scope_retained(self):
        record = raw_paper(2, title="Concrete prediction using Super Learner", abstract="")
        records, _ = fetch.dedupe_and_filter([("super learner", record)], min_year=2006,
                                            blocked={"paper_ids": set(), "dois": set(), "titles": set()})
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].matched_keywords, ["super learner"])

    def test_committed_data_generates_page(self):
        root = Path(__file__).resolve().parents[1]
        text = generate.generate_qmd(root / "data/semantic_scholar_tmle_papers.json",
                                     root / "data/blocked_tmle_papers.json")
        self.assertIn("Total tracked papers:", text)
        self.assertIn("Yearly Papers Trend", text)


if __name__ == "__main__":
    unittest.main()
