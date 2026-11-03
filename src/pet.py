"""Monthly PET on a lat/lon grid. Thornthwaite as in Vicente-Serrano et al. (2010); Hargreaves as in FAO-56."""
import numpy as np


def month_terms(time):
    """Mid-month day of year J and number of days NDM, for month-start time stamps."""
    ndm = time.dt.days_in_month
    J = time.dt.dayofyear + (ndm - 1) / 2
    return J, ndm


def daylength(lat, J):
    """N, maximum sunshine hours, from latitude (degrees) and day of year (paper's equations)."""
    phi = np.deg2rad(lat)
    delta = 0.4093 * np.sin(2 * np.pi * J / 365 - 1.405)
    ws = np.arccos(-np.tan(phi) * np.tan(delta))
    return 24 / np.pi * ws


def extraterrestrial_radiation(lat, J):
    """Ra in MJ m-2 day-1 (FAO-56 eqs. 21-25)."""
    phi = np.deg2rad(lat)
    dr = 1 + 0.033 * np.cos(2 * np.pi * J / 365)
    delta = 0.409 * np.sin(2 * np.pi * J / 365 - 1.39)
    ws = np.arccos(-np.tan(phi) * np.tan(delta))
    return (24 * 60 / np.pi) * 0.0820 * dr * (
        ws * np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.sin(ws))


def heat_index(tmean):
    """Thornthwaite heat index I for each cell, from its monthly climatology."""
    clim = tmean.groupby("time.month").mean("time")
    return ((clim.clip(min=0) / 5) ** 1.514).sum("month", min_count=1)


def thornthwaite(tmean, I=None):
    """Monthly PET in mm/month. Pass I to hold the heat index fixed (Phase 8)."""
    if I is None:
        I = heat_index(tmean)
    m = 6.75e-7 * I**3 - 7.71e-5 * I**2 + 1.79e-2 * I + 0.492
    J, ndm = month_terms(tmean.time)
    K = daylength(tmean.lat, J) / 12 * ndm / 30
    pet = 16 * K * (10 * tmean.clip(min=0) / I) ** m
    return pet.where(tmean.notnull())


def hargreaves(tmax, tmin):
    """Monthly reference ET in mm/month (Hargreaves and Samani 1985; FAO-56 eq. 52)."""
    J, ndm = month_terms(tmax.time)
    ra = 0.408 * extraterrestrial_radiation(tmax.lat, J)   # MJ m-2 d-1 to mm/day
    tmean = (tmax + tmin) / 2
    et0 = 0.0023 * ra * (tmean + 17.8) * np.sqrt((tmax - tmin).clip(min=0)) * ndm
    return et0.where(tmax.notnull())