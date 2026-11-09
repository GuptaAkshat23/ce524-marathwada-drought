"""Monotonic trends in annual series: Mann-Kendall test with Sen's slope.

For each series:
    slope  Sen's slope, units per year (median of all pairwise slopes)
    p      original Mann-Kendall p-value (two-sided)
    p_pw   p-value after pre-whitening (von Storch 1995): a conservative check
           against lag-1 autocorrelation, which makes the original test too liberal
    tau    Kendall's tau
    r1     lag-1 autocorrelation of the series after removing the Sen trend
Cells with fewer than MIN_YEARS values (outside Marathwada) are left missing.
"""
import numpy as np
import xarray as xr
import pymannkendall as mk

MIN_YEARS = 30
FIELDS = ["slope", "p", "p_pw", "tau", "r1"]


def _trend_1d(x):
    out = np.full(len(FIELDS), np.nan)
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < MIN_YEARS or np.ptp(x) == 0:
        return out
    r = mk.original_test(x)
    res = x - r.slope * np.arange(x.size)
    res = res - res.mean()
    r1 = (res[:-1] * res[1:]).sum() / (res ** 2).sum()
    out[:] = r.slope, r.p, mk.pre_whitening_modification_test(x).p, r.Tau, r1
    return out


def trend(da, dim="year"):
    """The FIELDS for every series along dim; adds a 'field' dimension."""
    out = xr.apply_ufunc(_trend_1d, da, input_core_dims=[[dim]], output_core_dims=[["field"]],
                         vectorize=True)
    return out.assign_coords(field=FIELDS)


def fdr(p, q=0.10):
    """Benjamini-Hochberg: True where a p-value stays significant at false discovery rate q.

    Applied to all cells of one map together (Wilks 2016 recommends q = 0.10 for
    spatially correlated fields). Missing p-values are never significant.
    """
    v = np.asarray(p, dtype=float)
    ok = np.isfinite(v)
    ps = np.sort(v[ok])
    passed = ps <= q * np.arange(1, ps.size + 1) / ps.size
    cut = ps[passed].max() if passed.any() else -1.0
    return xr.DataArray(ok & (v <= cut), coords=p.coords, dims=p.dims) if isinstance(p, xr.DataArray) \
        else ok & (v <= cut)