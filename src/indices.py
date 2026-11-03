"""SPI and SPEI on monthly series, following Vicente-Serrano et al. (2010).

SPI:  k-month rainfall sums, two-parameter gamma per calendar month, with a
      probability mass for zero totals (McKee et al. 1993).
SPEI: k-month sums of D = P - PET, three-parameter log-logistic per calendar
      month from L-moments, written in its generalized-logistic form so that it
      also fits months where D is not positively skewed.
Both are converted to standard normal values.
"""
import numpy as np
import xarray as xr
from scipy import stats

MIN_YEARS = 20       # fewer valid values than this in a calendar month: no fit
MIN_NONZERO = 10     # SPI: fewer non-zero totals than this: no fit (dry-season months)


def accumulate(x, k):
    """Sum over the current and previous k - 1 months; the first k - 1 months are NaN."""
    return x.rolling(time=k, min_periods=k).sum()


# ---------- L-moments and the log-logistic (SPEI) ----------

def lmoments(x):
    """l1, l2 and L-skewness t3 from unbiased probability-weighted moments."""
    x = np.sort(x)
    n = x.size
    i = np.arange(1, n + 1)
    b0 = x.mean()
    b1 = np.mean((i - 1) / (n - 1) * x)
    b2 = np.mean((i - 1) * (i - 2) / ((n - 1) * (n - 2)) * x)
    l1, l2, l3 = b0, 2 * b1 - b0, 6 * b2 - 6 * b1 + b0
    return l1, l2, l3 / l2


def loglogistic_fit(x):
    """Location xi, scale a, shape k of the generalized logistic (Hosking and Wallis 1997).

    For t3 > 0 this is the paper's log-logistic, with beta = 1/t3 = -1/k,
    alpha = beta * a and origin gamma = xi - alpha.
    """
    l1, l2, t3 = lmoments(x)
    k = -t3
    if abs(k) < 1e-6:                          # symmetric: plain logistic
        return l1, l2, 0.0
    a = l2 * np.sin(k * np.pi) / (k * np.pi)
    xi = l1 - a * (1 / k - np.pi / np.sin(k * np.pi))
    return xi, a, k


def loglogistic_cdf(x, xi, a, k):
    if k == 0:
        y = (x - xi) / a
    else:                                      # beyond the bound: F = 0 (k < 0) or 1 (k > 0)
        y = -np.log(np.maximum(1 - k * (x - xi) / a, 1e-12)) / k
    return 1 / (1 + np.exp(-y))


# ---------- gamma with zeros (SPI) ----------

def gamma_fit(x):
    """Probability of zero q, and gamma shape and scale fitted to the non-zero totals."""
    pos = x[x > 0]
    if pos.size < MIN_NONZERO:
        return None
    shape, _, scale = stats.gamma.fit(pos, floc=0)
    return 1 - pos.size / x.size, shape, scale


def gamma_cdf(x, q, shape, scale):
    F = q + (1 - q) * stats.gamma.cdf(x, shape, scale=scale)
    return np.where(x > 0, F, q / 2)           # zeros sit at the middle of their mass


# ---------- standardisation ----------

FITS = {"gamma": (gamma_fit, gamma_cdf), "loglogistic": (loglogistic_fit, loglogistic_cdf)}


def _standardise_1d(x, ref, month, kind):
    """One cell: fit each calendar month on ref, evaluate x, convert to z."""
    fit, cdf = FITS[kind]
    out = np.full(x.shape, np.nan)
    for m in range(1, 13):
        cal = ref[(month == m) & np.isfinite(ref)]
        sel = (month == m) & np.isfinite(x)
        if cal.size < MIN_YEARS:
            continue
        p = fit(cal)
        if p is None:
            continue
        F = np.clip(cdf(x[sel], *p), 1e-6, 1 - 1e-6)
        out[sel] = stats.norm.ppf(F)
    return out


def standardise(x, kind, ref=None):
    """Standardise a monthly DataArray cell by cell. ref is the calibration series (default x)."""
    ref = x if ref is None else ref
    return xr.apply_ufunc(
        _standardise_1d, x, ref, x.time.dt.month,
        input_core_dims=[["time"], ["time"], ["time"]], output_core_dims=[["time"]],
        vectorize=True, kwargs={"kind": kind},
    ).transpose(*x.dims)


def spi(pr, k, ref=None):
    return standardise(accumulate(pr, k), "gamma", None if ref is None else accumulate(ref, k))


def spei(d, k, ref=None):
    return standardise(accumulate(d, k), "loglogistic", None if ref is None else accumulate(ref, k))