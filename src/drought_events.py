"""Drought events from a monthly index by run theory (Yevjevich 1967).

An event is a run of consecutive months with the index at or below a threshold
(default -1, moderate drought). For each event:
    duration  D = number of months in the run
    severity  S = sum of |index| over the run (index values are negative)
    intensity I = S / D
    peak        = lowest index value in the run
A missing month ends a run. Events cut by the start or end of the record are
kept, so their duration and severity are lower limits.
"""
import numpy as np
import pandas as pd
import xarray as xr

THRESHOLD = -1.0
STATS = ["n_events", "events_per_decade", "mean_duration", "max_duration",
         "mean_severity", "max_severity", "mean_intensity", "pct_time"]


def runs(z, threshold=THRESHOLD):
    """Start and end positions (inclusive) of runs with z <= threshold."""
    below = np.asarray(z) <= threshold        # NaN compares False, so it ends a run
    edges = np.diff(np.concatenate(([0], below.astype(int), [0])))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1


def events(z, time, threshold=THRESHOLD):
    """Table of events for one series: start, end, duration, severity, intensity, peak."""
    z = np.asarray(z, dtype=float)
    rows = []
    for a, b in zip(*runs(z, threshold)):
        seg = z[a:b + 1]
        d, s = seg.size, -seg.sum()
        rows.append({"start": time[a], "end": time[b], "duration": d,
                     "severity": s, "intensity": s / d, "peak": seg.min()})
    return pd.DataFrame(rows, columns=["start", "end", "duration", "severity", "intensity", "peak"])


def _summary_1d(z, threshold):
    """The STATS values for one cell's series."""
    out = np.full(len(STATS), np.nan)
    valid = np.isfinite(z)
    if not valid.any():
        return out                                # cell outside Marathwada
    s, e = runs(z, threshold)
    d = e - s + 1
    sev = np.array([-z[a:b + 1].sum() for a, b in zip(s, e)])
    out[0] = s.size
    out[1] = s.size / (valid.sum() / 12) * 10
    out[7] = 100 * (z[valid] <= threshold).mean()
    if s.size:
        out[2:7] = d.mean(), d.max(), sev.mean(), sev.max(), (sev / d).mean()
    return out


def summarise(z, threshold=THRESHOLD):
    """Event statistics for every cell (and any other dimension) of a DataArray with a time axis."""
    out = xr.apply_ufunc(
        _summary_1d, z, input_core_dims=[["time"]], output_core_dims=[["stat"]],
        vectorize=True, kwargs={"threshold": threshold},
    )
    return out.assign_coords(stat=STATS)