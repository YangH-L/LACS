"""All E6 tests as one unittest module (used by the server's selftest stage):

    python -m unittest -v e6.tests.test_e6

It only collects the tests of test_units, test_pipeline and test_fetch_http (unittest's load_tests
protocol). Discovery (python -m pytest e6/tests, python -m unittest discover -s e6/tests -t .) finds
those modules directly and this one contributes nothing, so no test runs twice.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_tests(loader: unittest.TestLoader, tests: unittest.TestSuite, pattern):
    if pattern is not None:  # called by discovery: the other modules are discovered on their own
        return tests
    from e6.tests import test_fetch_hardening, test_fetch_http, test_pipeline, test_units

    suite = unittest.TestSuite()
    for mod in (test_units, test_pipeline, test_fetch_http, test_fetch_hardening):
        suite.addTests(loader.loadTestsFromModule(mod))
    return suite


if __name__ == "__main__":
    unittest.main()
