"""When the strategy should have worked but didn't.

The conditions report answers the easy question: which market states went
with outperformance. This module asks the harder one a Lucera judge raised:
when those states were present and the strategy still underperformed, what
was going on? If the failures are explained by an indicator we measure, the
conditions model is missing an interaction. If nothing we measure separates
them and they cluster late in history, something the data set does not
contain (crowding after publication, changes in how the market trades) has
drifted, and the strategy's edge may not be portable to the future.

Method, all deterministic:

1. Favorability score. Take the indicators the conditions report found
   notable for outperformance (|effect| >= MIN_EFFECT). Convert each to its
   percentile over history, centred to [-1, 1], weight by the signed effect,
   and average. +1 means every indicator sits where it sat on the strategy's
   best days. The score is smoothed with the same centred window the labels
   use, so it describes the same regime the label describes.
2. Expected vs. actual. Days in the top tercile of the score are "favorable",
   the bottom tercile "unfavorable". Cross-tab against the labels. Favorable
   and underperform = "should have worked but didn't"; unfavorable and
   outperform = "worked when it shouldn't have".
3. What was different about the failures, contrasted against the successes
   among favorable days only: the indicators the score did NOT use
   (rank-biserial, as in the conditions report; an indicator inside the score
   always looks different on failure days, because failures sit lower on the
   score by construction, so those are reported separately as the score gap),
   the strategy's own turnover and exposure, and time (rank-biserial of the
   dates: +1 means every failure came after every success), plus the hit rate
   by era and before/after publication.
4. A verdict chosen by fixed thresholds, so the AI layer narrates it rather
   than deciding it.

The score is fit on the full sample, including the failure periods, so the
hit rate is in-sample and flatters the conditions model. That makes the
failures more notable, not less: they are days the model could not fit even
with hindsight.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from regime import compare_conditions, rank_biserial

MIN_EFFECT = 0.2       # conditions-report effect size needed to enter the score
MIN_RUN = 10           # failure or surprise runs shorter than this (days) are not listed
MAX_PERIODS = 8        # longest runs listed
ERA_YEARS = 5
MIN_ERA_DAYS = 126     # an era with fewer favorable days than this is listed but not rated
MIN_FAILURE_DAYS = 63  # fewer failure days than this and the contrast is not meaningful
EXPLAINED_EFFECT = 0.4 # an indicator this different on failures counts as a measured explanation
DATE_EFFECT = 0.3      # failures this much later (or earlier) than successes count as clustering in time
MARGINAL_EFFECT = 0.4  # failures this much lower on the score than successes: conditions only marginally present
WEAK_LIFT = 1.15       # hit rate / base rate below this: the conditions barely predict anything
TURNOVER_RATIO = 1.5   # failure-vs-success turnover ratio this large counts as a behaviour difference


def _num(x) -> float | None:
    """Finite float, or None (JSON has no NaN)."""
    x = float(x)
    return x if np.isfinite(x) else None


def favorability_score(state: pd.DataFrame, conditions: list[dict], window: int) -> tuple[pd.Series, list[dict]]:
    """Daily score in [-1, 1] from the notable outperform indicators, smoothed over `window` days."""
    used = [c for c in conditions if abs(c["effect"]) >= MIN_EFFECT and c["indicator"] in state.columns]
    if not used:
        return pd.Series(np.nan, index=state.index), []
    contrib, weight = [], []
    for c in used:
        centred = 2 * state[c["indicator"]].rank(pct=True) - 1
        contrib.append(centred * c["effect"])
        weight.append(centred.notna() * abs(c["effect"]))
    num = pd.concat(contrib, axis=1).sum(axis=1, min_count=1)
    den = pd.concat(weight, axis=1).sum(axis=1)
    raw = num / den.replace(0, np.nan)
    score = raw.rolling(window, center=True, min_periods=max(1, window // 2)).mean()
    return score, [{"indicator": c["indicator"], "label": c["label"], "effect": c["effect"]} for c in used]


def _runs(mask: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp, int]]:
    """Contiguous True blocks of a boolean series on a trading-day index."""
    out = []
    change = mask.ne(mask.shift()).cumsum()
    for _, grp in mask[mask].groupby(change[mask]):
        out.append((grp.index[0], grp.index[-1], len(grp)))
    return out


def _period_rows(mask: pd.Series, strategy: pd.Series, benchmark: pd.Series, score: pd.Series) -> tuple[list[dict], int]:
    runs = [r for r in _runs(mask) if r[2] >= MIN_RUN]
    longest = sorted(sorted(runs, key=lambda r: -r[2])[:MAX_PERIODS])
    rows = []
    for start, end, days in longest:
        s, b = strategy.loc[start:end], benchmark.loc[start:end]
        sr, br = float((1 + s).prod() - 1), float((1 + b).prod() - 1)
        rows.append({"start": start.strftime("%Y-%m-%d"), "end": end.strftime("%Y-%m-%d"), "days": int(days),
                     "strategy_return": sr, "benchmark_return": br, "excess_return": sr - br,
                     "score": float(score.loc[start:end].mean())})
    return rows, len(runs)


def _rates(labels: pd.Series) -> dict:
    n = int(len(labels))
    return {"favorable_days": n,
            "hit_rate": float((labels == "outperform").mean()) if n else None,
            "miss_rate": float((labels == "underperform").mean()) if n else None}


def _verdict(score_indicators: list[dict], lift: float, n_fail: int, differed: list[dict],
             date_effect: float, eras: list[dict], behaviour: dict | None = None,
             score_gap: dict | None = None) -> dict:
    """Which findings pass the fixed thresholds, and what they mean together.

    `flags` lists every finding that fired; `kind` is the headline. The AI
    layer narrates this; it does not choose it.
    """
    if not score_indicators:
        return {"kind": "no_signal", "flags": [], "text": "No indicator was notably different on the strategy's "
                "outperforming days, so there is no set of favorable conditions to test against."}
    if lift < WEAK_LIFT:
        return {"kind": "weak_conditions", "flags": [], "text": f"The favorable conditions barely raise the odds of "
                f"outperforming (lift {lift:.2f}x over the base rate), so failures under them are not informative."}
    if n_fail < MIN_FAILURE_DAYS:
        return {"kind": "rare_failures", "flags": [], "text": f"The strategy underperformed on only {n_fail} favorable "
                f"days, too few to say what sets them apart."}

    flags, parts = [], []
    top = differed[0] if differed else None
    if top and abs(top["effect"]) >= EXPLAINED_EFFECT:
        flags.append("measured")
        parts.append(f"An indicator the conditions ignored separates the failures from the successes: {top['label']} was "
                     f"{'higher' if top['effect'] > 0 else 'lower'} on failure days (effect {top['effect']:+.2f}; median "
                     f"{top['inside_median']:.3g} vs {top['baseline_median']:.3g}). The conditions are missing this interaction.")
    else:
        parts.append(f"No indicator outside the conditions separates the failures from the successes"
                     + (f" (largest effect {top['effect']:+.2f} for {top['label']})." if top else "."))
    g = score_gap or {}
    if (g.get("effect") or 0.0) <= -MARGINAL_EFFECT:
        flags.append("marginal")
        parts.append(f"The failures came with weaker favorable readings than the successes (score {g['failure_mean']:+.2f} vs "
                     f"{g['success_mean']:+.2f}; effect {g['effect']:+.2f}): the conditions were only marginally present.")

    rated = [e for e in eras if e["rated"]]
    trend = (f" The hit rate went from {rated[0]['hit_rate']:.0%} in {rated[0]['era']} to {rated[-1]['hit_rate']:.0%} in {rated[-1]['era']}."
             if len(rated) >= 2 else "")
    date_effect = date_effect or 0.0
    if date_effect >= DATE_EFFECT:
        flags.append("later")
        parts.append(f"The failures cluster later in history (date effect {date_effect:+.2f}).{trend}")
    elif date_effect <= -DATE_EFFECT:
        flags.append("earlier")
        parts.append(f"The failures cluster early in history (date effect {date_effect:+.2f}).{trend}")
    else:
        parts.append(f"The failures do not clearly cluster in time (date effect {date_effect:+.2f}).{trend}")

    b = behaviour or {}
    tf, ts = b.get("turnover_failure") or 0.0, b.get("turnover_success") or 0.0
    if ts > 0 and tf / ts >= TURNOVER_RATIO:
        flags.append("traded_more")
        parts.append(f"The strategy traded {tf / ts:.1f}x more on failure days than on success days.")
    elif tf > 0 and ts / tf >= TURNOVER_RATIO:
        flags.append("traded_less")
        parts.append(f"The strategy traded {ts / tf:.1f}x less on failure days than on success days.")

    if "measured" in flags:
        kind = "measured"
        if "later" in flags:
            parts.append("Both a measured indicator and time separate the failures, so the decline may be that "
                         "indicator drifting, or something unmeasured; the data cannot say which.")
    elif "later" in flags:
        kind = "later"
        parts.append("That pattern is consistent with something this data set does not measure having changed over "
                     "time, such as crowding after publication or changes in how the market trades. The data cannot say which.")
    elif "earlier" in flags:
        kind = "earlier"
        parts.append("The conditions model fits the later data better than the earlier data.")
    elif "marginal" in flags:
        kind = "marginal"
        parts.append("The failures look like borderline cases rather than evidence of a hidden variable.")
    else:
        kind = "unexplained"
        parts.append("Whatever made the strategy fail under favorable conditions is not among the measured indicators, "
                     "and the failures do not clearly cluster in time.")
    return {"kind": kind, "flags": flags, "text": " ".join(parts)}


def expectation_check(labels: pd.Series, strategy: pd.Series, benchmark: pd.Series, weights: pd.DataFrame,
                      state: pd.DataFrame, conditions: list[dict], published: int | None, window: int) -> dict:
    """Cross-check favorable conditions against actual outcomes. See module docstring.

    `labels` are the per-day regime labels, `conditions` the outperform
    conditions report, and `window` the label window (days).
    """
    score, used = favorability_score(state, conditions, window)
    score = score.reindex(labels.index)
    valid = labels.notna() & score.notna()
    if not used or valid.sum() < 2 * window:
        return {"available": False, "score_indicators": used, "window": window,
                "verdict": _verdict(used, 0.0, 0, [], 0.0, [], {})}

    lab, sc = labels[valid], score[valid]
    lo, hi = sc.quantile([1 / 3, 2 / 3])
    favorable, unfavorable = sc >= hi, sc <= lo
    crosstab = {name: {k: int((lab[m] == k).sum()) for k in ("outperform", "match", "underperform")}
                for name, m in (("favorable", favorable), ("unfavorable", unfavorable), ("all", pd.Series(True, index=lab.index)))}
    base = float((lab == "outperform").mean())
    rates = _rates(lab[favorable])
    lift = rates["hit_rate"] / base if base > 0 else float("nan")

    failure = favorable & (lab == "underperform")
    success = favorable & (lab == "outperform")
    surprise = unfavorable & (lab == "outperform")
    failures, n_failure_runs = _period_rows(failure, strategy, benchmark, sc)
    surprises, n_surprise_runs = _period_rows(surprise, strategy, benchmark, sc)

    # Time: do the failures come later than the successes?
    date_effect = (_num(rank_biserial(pd.Series(lab.index[failure].asi8), pd.Series(lab.index[success].asi8)))
                   if failure.any() and success.any() else None)
    eras = []
    for era, grp in lab[favorable].groupby(lab.index[favorable].year // ERA_YEARS * ERA_YEARS):
        row = {"era": f"{era}-{era + ERA_YEARS - 1}", **_rates(grp)}
        row["rated"] = row["favorable_days"] >= MIN_ERA_DAYS
        eras.append(row)
    by_publication = None
    if published is not None:
        cut = pd.Timestamp(f"{published}-01-01")
        fav = lab[favorable]
        before, after = fav[fav.index < cut], fav[fav.index >= cut]
        by_publication = {"year": published,
                          "before": _rates(before) if len(before) >= MIN_ERA_DAYS else None,
                          "after": _rates(after) if len(after) >= MIN_ERA_DAYS else None}

    # Indicators, and the strategy's own behaviour, on failures vs. successes.
    in_score = {i["indicator"] for i in used}
    contrast = compare_conditions(failure.reindex(state.index, fill_value=False), state,
                                  universe=(failure | success).reindex(state.index, fill_value=False))
    differed = [d for d in contrast if d["indicator"] not in in_score]
    score_differed = [d for d in contrast if d["indicator"] in in_score]
    score_gap = {"effect": _num(rank_biserial(sc[failure], sc[success])) if failure.any() and success.any() else None,
                 "failure_mean": _num(sc[failure].mean()) if failure.any() else None,
                 "success_mean": _num(sc[success].mean()) if success.any() else None}
    turnover = weights.diff().abs().sum(axis=1).reindex(lab.index).fillna(0.0)
    gross = weights.abs().sum(axis=1).reindex(lab.index).fillna(0.0)
    behaviour = {"turnover_failure": _num(turnover[failure].mean()) if failure.any() else None,
                 "turnover_success": _num(turnover[success].mean()) if success.any() else None,
                 "exposure_failure": _num(gross[failure].mean()) if failure.any() else None,
                 "exposure_success": _num(gross[success].mean()) if success.any() else None}

    return {
        "available": True, "window": window, "score_indicators": used,
        "favorable_cut": float(hi), "unfavorable_cut": float(lo),
        "favorable_runs": [{"start": s.strftime("%Y-%m-%d"), "end": e.strftime("%Y-%m-%d")} for s, e, _ in _runs(favorable)],
        "crosstab": crosstab, "base_rate": base, "lift": _num(lift), **rates,
        "failure_days": int(failure.sum()), "success_days": int(success.sum()), "surprise_days": int(surprise.sum()),
        "failures": failures, "n_failure_runs": n_failure_runs,
        "surprises": surprises, "n_surprise_runs": n_surprise_runs,
        "date_effect": date_effect, "by_era": eras, "by_publication": by_publication,
        "what_differed": differed, "score_differed": score_differed, "score_gap": score_gap, "behaviour": behaviour,
        "verdict": _verdict(used, lift, int(failure.sum()), differed, date_effect, eras, behaviour, score_gap),
    }


def favorable_share(score: pd.Series, labels: pd.Series, cut: float, start, end) -> dict:
    """How favorable the conditions were inside one window, for the period explainer."""
    sc = score.loc[start:end].dropna()
    if sc.empty:
        return {"available": False}
    lab = labels.reindex(sc.index)
    return {"available": True, "score_mean": float(sc.mean()), "favorable_share": float((sc >= cut).mean()),
            "favorable_and_underperform_share": float(((sc >= cut) & (lab == "underperform")).mean())}
