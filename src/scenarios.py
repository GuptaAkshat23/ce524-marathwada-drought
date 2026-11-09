"""Warming scenarios (Phase 8): add a progressive temperature ramp, recompute PET, and re-standardise.

Rainfall is never changed, so SPI is identical in every scenario and serves as the control.
    th_fixed  Thornthwaite with the heat index I held at its observed value (primary)
    th_recalc Thornthwaite with I recomputed from the warmed series (as in the paper's setup)
    hg        Hargreaves from warmed Tmax and Tmin (diurnal range unchanged)
calib 'ref': fit on observed D, score warmed D.  calib 'self': fit on the warmed D (paper).
"""
import numpy as np
import xarray as xr
from . import pet, indices

VARIANTS = ("th_fixed", "th_recalc", "hg")


def ramp(time, total):
    """Linear warming in degC: 0 in the first month, `total` in the last."""
    return xr.DataArray(np.linspace(0.0, total, time.size), coords={"time": time}, dims="time")


def warm(da, total):
    return da + ramp(da.time, total)


def pet_variant(variant, tmean, tmax, tmin, I_obs):
    if variant == "th_fixed":
        return pet.thornthwaite(tmean, I=I_obs)
    if variant == "th_recalc":
        return pet.thornthwaite(tmean)
    if variant == "hg":
        return pet.hargreaves(tmax, tmin)
    raise ValueError(variant)


def d_series(variant, ds, total=0.0):
    """P - PET (mm/month) for a variant, with `total` degC of progressive warming."""
    I_obs = pet.heat_index(ds["tmean"])
    tmean, tmax, tmin = (warm(ds[v], total) for v in ("tmean", "tmax", "tmin"))
    return ds["pr"] - pet_variant(variant, tmean, tmax, tmin, I_obs)


def spei_scenario(variant, ds, total, k, calib="ref", reduce=None):
    """SPEI at scale k. reduce(da) turns cells into a regional series."""
    d_obs, d_w = d_series(variant, ds, 0.0), d_series(variant, ds, total)
    if reduce is not None:
        d_obs, d_w = reduce(d_obs), reduce(d_w)
    return indices.spei(d_w, k, ref=d_obs if calib == "ref" else None)


def december(z):
    """One calendar-year value per year: December of a 12-month index."""
    z = z.where(z.time.dt.month == 12, drop=True)
    return z.assign_coords(year=z.time.dt.year).swap_dims(time="year").drop_vars("time")


def corr_by_scale(a_by_k, b_by_k):
    """Pearson r between two dicts {k: monthly series}, over months where both exist."""
    out = {}
    for k in a_by_k:
        a, b = a_by_k[k].values, b_by_k[k].values
        ok = np.isfinite(a) & np.isfinite(b)
        out[k] = float(np.corrcoef(a[ok], b[ok])[0, 1])
    return out


def saturated(z, cut=-3.0):
    """Share of months at or below `cut` (a large share means the index is saturating)."""
    v = np.asarray(z)
    v = v[np.isfinite(v)]
    return float((v <= cut).mean())