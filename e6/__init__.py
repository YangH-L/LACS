"""E6: US Congress Twitter network + DW-NOMINATE (pre-registered plan, docs/PLANS.md (lacs_e6_plan_20261025)).

Stages (run from the package root, the directory that contains lacs/ and e6/):

  python -m e6.fetch     --data DATA_DIR                      # e6fetch: Voteview + GovInfo BILLSTATUS
  python -m e6.build     --data DATA_DIR --out RUN_DIR        # e6build: tables, tolerances, geo dataset
  python -m e6.structure --run RUN_DIR --per-query CSV        # plan section 4
  python -m e6.premise   --run RUN_DIR                        # plan section 5 (Q1)
  python -m e6.static    --groups P0_EXPORT_DIR --results DIR # plan section 6 (Q2)
  python -m e6.report    --run RUN_DIR                        # E6_REPORT.md + e6_report.json

Sections 7 (dynamics, Q3) and 8 (search outcome, Q4) are run by other code; build writes their
inputs (RUN_DIR/geo/*, RUN_DIR/graph/graph.bin) and report includes their results when present.
"""

__version__ = "0.1.0"
