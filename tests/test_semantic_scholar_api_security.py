"""Offline, synthetic-credential coverage. Never load a real API key."""
import io
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import ssl
import sys
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import fetch_semantic_scholar_papers as fetch
import generate_tmle_review_summary as review

KEY = 'FAKE-TEST-KEY+/no-real-credential'


class ApiSecurityTests(unittest.TestCase):
    def call(self, **overrides):
        return fetch._http_get_json(**dict(
            url=fetch.API_BULK_URL + '?query=tmle', headers={'x-api-key': KEY},
            ssl_context=None, timeout_s=1, max_retries=0,
            backoff_s=2, max_backoff_s=10, **overrides))

    def response(self, data):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps(data).encode()
        return response

    def test_key_sent_in_header_not_url(self):
        with patch.object(fetch, '_open_response', return_value=self.response({'data': []})) as op:
            self.assertEqual(self.call(), {'data': []})
        req = op.call_args.args[0]
        self.assertEqual(req.get_header('X-api-key'), KEY)
        self.assertNotIn(KEY, req.full_url)

    def test_http_failure_omits_raw_body_headers_url_and_chaining(self):
        for code in (400, 401, 403, 429, 500):
            with self.subTest(code=code):
                err = urllib.error.HTTPError('https://example.test/'+KEY, code, KEY,
                                              {'X-Debug': KEY}, io.BytesIO(KEY.encode()))
                log = io.StringIO()
                with patch.object(fetch, '_open_response', side_effect=err), patch.object(sys, 'stderr', log):
                    with self.assertRaises(fetch.SafeDiscoveryError) as caught:
                        self.call()
                self.assertEqual(str(caught.exception), f'HTTP {code} from Semantic Scholar API')
                self.assertTrue(caught.exception.__suppress_context__)
                self.assertNotIn(KEY, log.getvalue())

    def test_network_error_retry_does_not_echo_exception(self):
        log = io.StringIO()
        with patch.object(fetch, '_open_response', side_effect=urllib.error.URLError(KEY)), \
             patch.object(sys, 'stderr', log), patch.object(fetch.time, 'sleep'):
            with self.assertRaises(fetch.SafeDiscoveryError) as caught:
                fetch._http_get_json(url=fetch.API_BULK_URL, headers={'x-api-key': KEY},
                    ssl_context=None, timeout_s=1, max_retries=1, backoff_s=2, max_backoff_s=10)
        self.assertNotIn(KEY, log.getvalue() + str(caught.exception))
        self.assertIn('retrying', log.getvalue())

    def test_reflected_credential_in_success_payload_is_rejected(self):
        for value in (KEY, urllib.parse.quote(KEY, safe=''), urllib.parse.quote_plus(KEY),
                      KEY.replace('+', '%2b').replace('/', '%2f'),
                      KEY.replace('+', '%2B').replace('/', '%2f')):
            with self.subTest(value=value), patch.object(fetch, '_open_response',
                    return_value=self.response({'data': [{'title': value}]})):
                with self.assertRaisesRegex(fetch.SafeDiscoveryError, 'discarded'):
                    self.call()

    def test_no_redirect_including_same_origin(self):
        req = urllib.request.Request(fetch.API_BULK_URL, headers={'x-api-key': KEY})
        for target in (fetch.API_URL, 'https://evil.test', 'http://api.semanticscholar.org/'):
            with self.subTest(target=target), self.assertRaisesRegex(fetch.SafeDiscoveryError, 'redirect refused'):
                fetch.NoRedirects().redirect_request(req, None, 302, KEY, {}, target)

    def test_real_opener_blocks_redirect_without_followup_request(self):
        paths = []
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                paths.append(self.path)
                self.send_response(302)
                self.send_header('Location', '/would-receive-key')
                self.end_headers()
            def log_message(self, *args):
                pass
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            request = urllib.request.Request(
                f'http://127.0.0.1:{server.server_port}/start', headers={'x-api-key': KEY})
            with self.assertRaisesRegex(fetch.SafeDiscoveryError, 'redirect refused'):
                fetch._open_response(request, timeout=1, context=None)
            self.assertEqual(paths, ['/start'])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_transport_installs_redirect_blocker_and_verified_tls(self):
        req = urllib.request.Request(fetch.API_BULK_URL)
        with patch.object(fetch.urllib.request, 'build_opener') as build:
            fetch._open_response(req, timeout=1, context=None)
        handlers = build.call_args.args
        self.assertIsInstance(handlers[0], fetch.NoRedirects)
        self.assertIsInstance(handlers[1], urllib.request.HTTPSHandler)
        self.assertEqual(handlers[1]._context.verify_mode, ssl.CERT_REQUIRED)
        self.assertTrue(handlers[1]._context.check_hostname)
        build.return_value.open.assert_called_once_with(req, timeout=1)

    def test_keyed_insecure_mode_rejected_before_fetch(self):
        with patch.dict(os.environ, {'SEMANTIC_SCHOLAR_API_KEY': KEY}), \
             patch.object(sys, 'argv', ['fetch', '--insecure']), \
             patch.object(fetch, 'fetch_keyword_bulk') as http:
            with self.assertRaisesRegex(SystemExit, 'not allowed'):
                fetch.main()
        http.assert_not_called()

    def test_untrusted_failure_never_reaches_log_report_or_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output, report, block = (root / name for name in ('papers.json', 'report.json', 'blocked.json'))
            output.write_text('{"papers": []}')
            original = output.read_bytes()
            block.write_text('{}')
            log = io.StringIO()
            argv = ['fetch', '--output', str(output), '--report', str(report),
                    '--blocked-json', str(block), '--keywords', 'tmle']
            with patch.dict(os.environ, {'SEMANTIC_SCHOLAR_API_KEY': KEY}), \
                 patch.object(sys, 'argv', argv), patch.object(sys, 'stderr', log), \
                 patch.object(fetch, 'fetch_keyword_bulk', side_effect=ValueError(KEY)):
                self.assertEqual(fetch.main(), 2)
            self.assertEqual(output.read_bytes(), original)
            review.main(['--baseline', str(output), '--catalog', str(output), '--report', str(report),
                         '--output-dir', str(root / 'diagnostics'), '--run-url', 'https://example.test/run',
                         '--step-summary', str(root/'step-summary.md')])
            self.assertNotIn(KEY, log.getvalue())
            for path in root.rglob('*'):
                if path.is_file():
                    self.assertNotIn(KEY, path.read_text())
            self.assertTrue(json.loads(report.read_text())['incomplete'])

    def test_http_failure_is_safe_through_complete_report_summary_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output, report, block = (root / name for name in ('papers.json', 'report.json', 'blocked.json'))
            output.write_text('{"papers": []}')
            block.write_text('{}')
            error = urllib.error.HTTPError(fetch.API_URL, 401, KEY, {}, io.BytesIO(KEY.encode()))
            log = io.StringIO()
            argv = ['fetch', '--output', str(output), '--report', str(report),
                    '--blocked-json', str(block), '--keywords', 'tmle']
            with patch.dict(os.environ, {'SEMANTIC_SCHOLAR_API_KEY': KEY}), \
                 patch.object(sys, 'argv', argv), patch.object(sys, 'stderr', log), \
                 patch.object(fetch, '_open_response', side_effect=error):
                self.assertEqual(fetch.main(), 2)
            review.main(['--baseline', str(output), '--catalog', str(output), '--report', str(report),
                         '--output-dir', str(root / 'diagnostics'), '--run-url', 'https://example.test/run',
                         '--step-summary', str(root/'step-summary.md')])
            self.assertIn('HTTP 401', log.getvalue())
            self.assertNotIn(KEY, log.getvalue())
            for path in root.rglob('*'):
                if path.is_file():
                    self.assertNotIn(KEY, path.read_text())

    def test_secret_is_scoped_to_fetch_step_and_missing_key_stops_before_python(self):
        workflow = (Path(__file__).resolve().parents[1] / '.github/workflows/tmle-papers.yml').read_text()
        start = workflow.index('      - name: Discover papers')
        end = workflow.index('      - name: Record discovery', start)
        step = workflow[start:end]
        self.assertIn('SEMANTIC_SCHOLAR_API_KEY: ${{ secrets.SEMANTIC_SCHOLAR_API_KEY }}', step)
        self.assertNotIn('secrets.SEMANTIC_SCHOLAR_API_KEY', workflow[:start]+workflow[end:])
        self.assertLess(step.index('exit 1'), step.index('python3 scripts/fetch'))
        self.assertIn('--request-interval-s 2', step)
        self.assertNotIn('--api-key', step)


if __name__ == '__main__':
    unittest.main()
