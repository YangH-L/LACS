"""e6.fetch's online path against a local HTTP server that mimics GovInfo and Voteview: XML and JSON
listings (the XML listing failing for one type), a bulk package plus a file outside it, a transient
503, a truncated response (well-formedness check, retried), the Voteview HS117 file missing (HSall
fallback, filtered to congress 117), a file that is permanently missing (exit 2), with requests and with
urllib. No external network access.
"""
from __future__ import annotations

import io
import json
import os
import shutil
import sys
import tempfile
import threading
import unittest
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e6 import fetch  # noqa: E402
from e6.tests.fixture import VV_COLS, _listing_xml, _xml_bill  # noqa: E402


def _vv_csv() -> bytes:
    rows = [",".join(VV_COLS)]
    for congress, b in ((116, "A000001"), (117, "A000001"), (117, "A000002")):
        r = {c: "" for c in VV_COLS}
        r.update({"congress": congress, "chamber": "House", "icpsr": 1, "state_abbrev": "NY", "party_code": 100,
                  "bioname": "X", "bioguide_id": b, "nominate_dim1": -0.3, "nominate_dim2": 0.1,
                  "nominate_number_of_votes": 100})
        rows.append(",".join(str(r[c]) for c in VV_COLS))
    return ("\n".join(rows) + "\n").encode()


class _Handler(BaseHTTPRequestHandler):
    files: dict = {}
    fail_once: set = set()
    truncate_once: set = set()
    always_fail: set = set()
    seen: dict = {}
    lock = threading.Lock()

    def log_message(self, *a):  # quiet
        pass

    def do_GET(self):  # noqa: N802
        p = self.path
        with self.lock:
            n = self.seen.get(p, 0)
            self.seen[p] = n + 1
        if p in self.always_fail:
            self.send_response(500)
            self.end_headers()
            return
        if p in self.fail_once and n == 0:
            self.send_response(503)
            self.end_headers()
            return
        body = self.files.get(p)
        if body is None:
            self.send_response(404)
            self.end_headers()
            return
        if p in self.truncate_once and n == 0:
            body = body[: len(body) // 2]
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class TestFetchHttp(unittest.TestCase):
    def setUp(self):
        # the local server must not be reached through a proxy configured on the machine
        self._env = {k: os.environ.get(k) for k in ("NO_PROXY", "no_proxy")}
        for k in self._env:
            os.environ[k] = ",".join(x for x in ("127.0.0.1", "localhost", self._env[k]) if x)
        self.tmp = Path(tempfile.mkdtemp(prefix="e6_http_"))
        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.base = f"http://127.0.0.1:{self.srv.server_address[1]}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        b = self.base
        files = {"/static/data/out/members/HSall_members.csv": _vv_csv()}
        xmls = {}
        for t, n in (("hr", 5), ("s", 3), ("hjres", 2), ("sjres", 2)):
            for k in range(1, n + 1):
                name = f"BILLSTATUS-117{t}{k}.xml"
                xmls[(t, name)] = _xml_bill("old" if k % 2 else "new", t, k, "2021-03-01", "A000001",
                                            [("A000002", "2021-03-02", "False", None)], []).encode()
                files[f"/bulkdata/BILLSTATUS/117/{t}/{name}"] = xmls[(t, name)]

        def listing(t, names):
            return _listing_xml(names, t).replace("https://www.govinfo.gov", b).encode()

        files["/bulkdata/xml/BILLSTATUS/117/hr"] = listing("hr", [f"BILLSTATUS-117hr{k}.xml" for k in range(1, 6)])
        # s: XML listing fails, JSON listing has a package (2 of the 3 files) and the 3 files
        zbuf = io.BytesIO()
        with zipfile.ZipFile(zbuf, "w") as z:
            for k in (1, 2):
                z.writestr(f"BILLSTATUS-117-s/BILLSTATUS-117s{k}.xml", xmls[("s", f"BILLSTATUS-117s{k}.xml")])
        files["/bulkdata/BILLSTATUS/117/s/BILLSTATUS-117-s.zip"] = zbuf.getvalue()
        files["/bulkdata/json/BILLSTATUS/117/s"] = json.dumps({"files": [
            {"justFileName": n, "fileExtension": n.rsplit(".", 1)[1], "folder": False,
             "link": f"{b}/bulkdata/BILLSTATUS/117/s/{n}"}
            for n in ["BILLSTATUS-117-s.zip"] + [f"BILLSTATUS-117s{k}.xml" for k in (1, 2, 3)]]}).encode()
        files["/bulkdata/xml/BILLSTATUS/117/hjres"] = listing("hjres", ["BILLSTATUS-117hjres1.xml",
                                                                        "BILLSTATUS-117hjres2.xml"])
        files["/bulkdata/xml/BILLSTATUS/117/sjres"] = listing("sjres", ["BILLSTATUS-117sjres1.xml",
                                                                        "BILLSTATUS-117sjres2.xml"])
        _Handler.files = files
        _Handler.seen = {}
        _Handler.always_fail = {"/bulkdata/xml/BILLSTATUS/117/s"}
        _Handler.fail_once = {"/bulkdata/BILLSTATUS/117/hjres/BILLSTATUS-117hjres2.xml"}
        _Handler.truncate_once = {"/bulkdata/BILLSTATUS/117/sjres/BILLSTATUS-117sjres1.xml"}
        self.args = ["--govinfo-url", self.base, "--voteview-url", self.base, "--retries", "2", "--backoff", "0.01",
                     "--threads", "3", "--timeout", "10"]

    def tearDown(self):
        self.srv.shutdown()
        self.srv.server_close()
        shutil.rmtree(self.tmp, ignore_errors=True)
        for k, v in self._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def _check_complete(self, data: Path):
        man = json.loads((data / "fetch_manifest.json").read_text())
        self.assertTrue(man["complete"], man)
        vv = man["voteview"]
        self.assertEqual(vv["derived_from"], "HSall_members.csv")
        self.assertEqual(vv["filtered_rows"], 2)  # congress 117 rows only
        self.assertEqual(vv["check"]["rows"], 2)
        bs = man["billstatus"]
        self.assertEqual(bs["hr"]["obtained_xml"], 5)
        self.assertEqual(bs["s"]["listing"]["format"], "json")
        self.assertEqual((bs["s"]["from_zip"], bs["s"]["downloaded"]), (2, 1))
        self.assertEqual(bs["hjres"]["obtained_xml"], 2)
        self.assertEqual(bs["sjres"]["obtained_xml"], 2)
        for t, n in (("hr", 5), ("s", 3), ("hjres", 2), ("sjres", 2)):
            self.assertEqual(len(list((data / "billstatus" / t).glob(f"BILLSTATUS-117{t}*.xml"))), n)
        self.assertFalse(list(data.rglob("*.part")))
        # the transient 503 and the truncated body were retried
        self.assertGreaterEqual(_Handler.seen["/bulkdata/BILLSTATUS/117/hjres/BILLSTATUS-117hjres2.xml"], 2)
        self.assertGreaterEqual(_Handler.seen["/bulkdata/BILLSTATUS/117/sjres/BILLSTATUS-117sjres1.xml"], 2)

    def test_online_requests(self):
        if fetch.requests is None:
            self.skipTest("requests not installed")
        data = self.tmp / "data"
        self.assertEqual(fetch.main(["--data", str(data)] + self.args), 0)
        self._check_complete(data)
        man = json.loads((data / "fetch_manifest.json").read_text())
        self.assertEqual(man["client"], "requests")

    def test_online_urllib_and_missing_file(self):
        saved = fetch.requests
        fetch.requests = None
        try:
            data = self.tmp / "data_urllib"
            self.assertEqual(fetch.main(["--data", str(data)] + self.args), 0)
            self._check_complete(data)
            self.assertEqual(json.loads((data / "fetch_manifest.json").read_text())["client"], "urllib")
            # a listed file that the server does not have: incomplete, exit 2
            del _Handler.files["/bulkdata/BILLSTATUS/117/hr/BILLSTATUS-117hr4.xml"]
            data2 = self.tmp / "data_missing"
            self.assertEqual(fetch.main(["--data", str(data2), "--types", "hr"] + self.args), 2)
            man = json.loads((data2 / "fetch_manifest.json").read_text())
            self.assertEqual(man["billstatus"]["hr"]["missing"], ["BILLSTATUS-117hr4.xml"])
            self.assertIn("404", man["billstatus"]["hr"]["failures"][0]["error"])
        finally:
            fetch.requests = saved


if __name__ == "__main__":
    unittest.main()
