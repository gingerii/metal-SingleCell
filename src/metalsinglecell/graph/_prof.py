"""Opt-in, zero-overhead profiling hooks for the clustering inner loops.

When ``ENABLED`` is False (the default) every hook is a cheap attribute read, so
the shipped clusterer pays nothing. A profiling driver flips ``ENABLED = True``,
calls :func:`reset`, runs the clusterer, then reads :data:`records` — a list of
per-level dicts (move/refine/contract wall time, pass counts, host-sync counts).

Pass counts flow through the module globals ``last_move_passes`` /
``last_refine_passes`` (set by the sync movers just before they return) rather
than through return signatures, so the instrumentation stays out of the API.
"""
from __future__ import annotations

ENABLED = False
records: list[dict] = []      # per-level records, appended by _leiden_pass
last_move_passes = 0          # set by _local_moving_sync / _local_moving
last_refine_passes = 0        # set by _refine_sync / _refine
last_move_syncs = 0           # GPU->host round-trips in the last move call
last_refine_syncs = 0         # GPU->host round-trips in the last refine call


def reset() -> None:
    global last_move_passes, last_refine_passes, last_move_syncs, last_refine_syncs
    records.clear()
    last_move_passes = last_refine_passes = 0
    last_move_syncs = last_refine_syncs = 0
