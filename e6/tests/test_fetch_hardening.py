"""e6.fetch hardening (review 10-25): explicit Accept headers on the listings (a server answering 406 to
'*/*'), the circuit breaker on systematic download failures, and the bill-number coverage check against
truncated listings. Local HTTP server only."""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e6 import fetch  # noqa: E402
from e6.tests.fixture import _listing_xml, _xml_bill  # noqa: E402
from e6.tests.test_fetch_http import _vv_csv  # noqa: E402


class _H(BaseHTTPRequestHandler):
    files: dict = {}
    listing_needs_xml_accept = False
    fail_files = False
    attempts: list = []
    lock = threading.Lock()

    def log_message(self, *a):
        pass

    def do_GET(self):  # noqa: N802
        p = self.path
        if p.startswith("/bulkdata/xml/") and self.listing_needs_xml_accept \
                and self.headers.get("Accept") != "application/xml":
            self.send_response(406)
            self.end_headers()
            return
        if p.startswith("/bulkdata/BILLSTATUS/") and self.fail_files:
            with self.lock:
                self.attempts.append(p)
            self.send_response(500)
            self.end_headers()
            return
        body = self.files.get(p)
        if body is None:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class TestFetchHardening(unittest.TestCase):
    def setUp(self):
        self._env = {k: os.environ.get(k) for k in ("NO_PROXY", "no_proxy")}
        for k in self._env:
            os.environ[k] = ",".join(x for x in ("127.0.0.1", "localhost", self._env[k]) if x)
        self.tmp = Path(tempfile.mkdtemp(prefix="e6_hard_"))
        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), _H)
        self.base = f"http://127.0.0.1:{self.srv.server_address[1]}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        _H.files = {"/static/data/out/members/HSall_members.csv": _vv_csv()}
        _H.listing_needs_xml_accept = False
        _H.fail_files = False
        _H.attempts = []
        self.args = ["--govinfo-url", self.base, "--voteview-url", self.base, "--retries", "0", "--backoff", "0.01",
                     "--threads", "2", "--timeout", "10", "--types", "hr"]

    def tearDown(self):
        self.srv.shutdown()
        self.srv.server_close()
        shutil.rmtree(self.tmp, ignore_errors=True)
        for k, v in self._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def _serve(self, numbers):
        names = [f"BILLSTATUS-117hr{k}.xml" for k in numbers]
        for k, n in zip(numbers, names):
            _H.files[f"/bulkdata/BILLSTATUS/117/hr/{n}"] = _xml_bill(
                "new", "hr", k, "2021-03-01", "A000001", [("A000002", "2021-03-02", "False", None)], []).encode()
        _H.files["/bulkdata/xml/BILLSTATUS/117/hr"] = _listing_xml(names, "hr").replace(
            "https://www.govinfo.gov", self.base).encode()

    def test_listing_with_xml_accept(self):
        self._serve(range(1, 6))
        _H.listing_needs_xml_accept = True  # 406 for */*, as GovInfo warns
        data = self.tmp / "d1"
        self.assertEqual(fetch.main(["--data", str(data)] + self.args), 0)
        man = json.loads((data / "fetch_manifest.json").read_text())
        self.assertTrue(man["complete"])
        self.assertEqual(man["billstatus"]["hr"]["obtained_xml"], 5)

    def test_circuit_breaker(self):
        self._serve(range(1, 301))
        _H.fail_files = True
        data = self.tmp / "d2"
        t0 = time.time()
        self.assertEqual(fetch.main(["--data", str(data)] + self.args), 2)
        self.assertLess(time.time() - t0, 60)
        rec = json.loads((data / "fetch_manifest.json").read_text())["billstatus"]["hr"]
        self.assertFalse(rec["complete"])
        self.assertTrue(any(f["name"] == "<type>" for f in rec["failures"]))
        self.assertLess(len(set(_H.attempts)), 300)  # it stopped early

    def test_low_number_coverage_is_incomplete(self):
        self._serve([1, 50, 100])
        data = self.tmp / "d3"
        self.assertEqual(fetch.main(["--data", str(data)] + self.args), 2)
        rec = json.loads((data / "fetch_manifest.json").read_text())["billstatus"]["hr"]
        self.assertFalse(rec["complete"])
        self.assertAlmostEqual(rec["number_coverage"]["coverage"], 0.03)


if __name__ == "__main__":
    unittest.main()
