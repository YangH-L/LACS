"""LACS P0 kit: interaction stress test (E1), sigma sweep (E2), exact baselines (E3)
and reporting fixes (E4) for the LCC paper."""

import os as _os

# Single-threaded BLAS/OpenMP in every P0 process, set before numpy is imported:
#  * the drivers parallelise over processes (8 workers x 14 BLAS threads would oversubscribe);
#  * a thread pool that MKL/OpenMP starts in a parent does not survive fork(), and a worker
#    that then multiplies matrices can wait forever (fork-safety);
#  * E3 timings stay single-threaded. Override by exporting the variables yourself.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")

# Diagnostics: `kill -USR1 <pid>` (or `pkill -USR1 -f p0kit`) writes the Python stack of every
# thread of a P0 process to its stderr, i.e. into the logs; handy if anything ever looks stuck.
try:
    import faulthandler as _fh
    import signal as _sig
    import sys as _sys

    if hasattr(_sig, "SIGUSR1"):
        _fh.register(_sig.SIGUSR1, file=_sys.stderr, all_threads=True)
except Exception:  # pragma: no cover
    pass

__version__ = "0.5.3-p0"
