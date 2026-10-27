#!/usr/bin/env bash
# Unit / property tests (CPU, ~1-2 minutes). Run from the project root.
set -euo pipefail
python -m lacs build-cpp
python -m unittest tests.test_data tests.test_search tests.test_sjtlat tests.test_dynamics tests.test_pipeline tests.test_geo tests.test_e3 -v
