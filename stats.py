"""The two statistical routines the app needs, in plain numpy.

These replace statsmodels (and with it scipy), which together were ~185 MB of
the deployed bundle for two function calls. Each routine reproduces the
statsmodels computation step for step; `tests/test_stats_parity.py` checks the
results against statsmodels itself when it is installed (dev only).

* `ols_hac`: OLS with Newey-West (HAC) standard errors, Bartlett kernel, no
  small-sample correction. Matches `sm.OLS(...).fit(cov_type="HAC",
  cov_kwds={"maxlags": L})`.
* `engle_granger`: two-variable Engle-Granger cointegration test with a
  constant, ADF lag chosen by AIC, MacKinnon (2010) asymptotic p-value.
  Matches `statsmodels.tsa.stattools.coint(y0, y1)`.
"""

import math

import numpy as np


def _ols(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Coefficients, residuals and (X'X)^-1, via the pseudo-inverse like statsmodels."""
    pinv = np.linalg.pinv(X)
    beta = pinv @ y
    return beta, y - X @ beta, pinv @ pinv.T


def ols_hac(y, X, maxlags: int) -> tuple[np.ndarray, np.ndarray]:
    """OLS coefficients and Newey-West t-statistics.

    Covariance = (X'X)^-1 S (X'X)^-1 with S = sum_l w_l (G_l + G_l'), where
    G_l = sum_t (x_t u_t)(x_{t-l} u_{t-l})' and Bartlett weights
    w_l = 1 - l / (maxlags + 1).
    """
    y, X = np.asarray(y, float), np.asarray(X, float)
    beta, resid, xtx_inv = _ols(y, X)
    xu = X * resid[:, None]
    S = xu.T @ xu
    for lag in range(1, maxlags + 1):
        g = xu[lag:].T @ xu[:-lag]
        S += (1 - lag / (maxlags + 1.0)) * (g + g.T)
    cov = xtx_inv @ S @ xtx_inv.T
    return beta, beta / np.sqrt(np.diag(cov))


def _aic_no_constant(y: np.ndarray, X: np.ndarray) -> float:
    """statsmodels OLS AIC for a regression without a constant: -2 llf + 2 k."""
    n = len(y)
    _, resid, _ = _ols(y, X)
    ssr = float(resid @ resid)
    llf = -n / 2 * (math.log(2 * math.pi) + math.log(ssr / n) + 1)
    return -2 * llf + 2 * np.linalg.matrix_rank(X)


def _adf_design(x: np.ndarray, lags: int) -> tuple[np.ndarray, np.ndarray]:
    """Regress dx_t on [x_{t-1}, dx_{t-1}, ..., dx_{t-lags}] (no constant)."""
    dx = np.diff(x)
    n = len(dx) - lags
    cols = [x[lags: lags + n]] + [dx[lags - k: lags - k + n] for k in range(1, lags + 1)]
    return dx[lags:], np.column_stack(cols)


def adf_stat(x, maxlag: int | None = None) -> float:
    """ADF t-statistic, no deterministic terms, lag chosen by AIC (as statsmodels)."""
    x = np.asarray(x, float)
    nobs = len(x)
    if maxlag is None:
        maxlag = min(nobs // 2 - 1, int(math.ceil(12.0 * (nobs / 100.0) ** 0.25)))
    # Lag search on a common sample (the one trimmed for maxlag).
    y_full, X_full = _adf_design(x, maxlag)
    best = min((_aic_no_constant(y_full, X_full[:, : lag + 1]), lag) for lag in range(maxlag + 1))[1]
    # Re-estimate with the chosen lag on its own (longer) sample.
    y, X = _adf_design(x, best)
    beta, resid, xtx_inv = _ols(y, X)
    sigma2 = float(resid @ resid) / (len(y) - X.shape[1])
    return float(beta[0] / math.sqrt(sigma2 * xtx_inv[0, 0]))


# MacKinnon (2010) response-surface coefficients for the asymptotic p-value,
# Engle-Granger with a constant and N = 2 variables (as tabulated in statsmodels).
_MK_MAX, _MK_MIN, _MK_STAR = 0.92, -18.86, -2.62
_MK_SMALL = (2.92, 1.5012, 0.039796)
_MK_LARGE = (2.1945, 0.64695, -0.29198, -0.042377)


def mackinnon_p(stat: float) -> float:
    if stat > _MK_MAX:
        return 1.0
    if stat < _MK_MIN:
        return 0.0
    coef = _MK_SMALL if stat <= _MK_STAR else _MK_LARGE
    z = sum(c * stat ** i for i, c in enumerate(coef))
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def engle_granger(y0, y1) -> tuple[float, float]:
    """(ADF t-statistic, p-value) for cointegration of y0 on y1 with a constant."""
    y0, y1 = np.asarray(y0, float), np.asarray(y1, float)
    X = np.column_stack([y1, np.ones_like(y1)])
    _, resid, _ = _ols(y0, X)
    r2 = 1 - (resid @ resid) / float(((y0 - y0.mean()) ** 2).sum())
    if r2 >= 1 - 100 * math.sqrt(np.finfo(float).eps):
        stat = -math.inf  # (almost) perfectly collinear; statsmodels does the same
    else:
        stat = adf_stat(resid)
    return stat, mackinnon_p(stat)
