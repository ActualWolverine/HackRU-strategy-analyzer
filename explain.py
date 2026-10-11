"""Explanations of backtest results, in plain language.

Design, following the Lucera guidance on bounded AI assistance:

* Every number comes from the engine. This module turns engine output into a
  compact table of *facts*, each with a key, and the explainer may only narrate
  those facts. It never sees prices, trades or any raw data.
* Each section of an explanation lists the fact keys it relies on. Citations to
  keys that don't exist are removed and the section is flagged, so AI output
  can't smuggle in unsupported claims unnoticed.
* The explainer is behind an interface. `GeminiExplainer` calls the Gemini API;
  `TemplateExplainer` builds the same structure deterministically from the facts
  and is used whenever no API key is configured or the API fails. Judges can run
  everything offline.
* Model latency is reported separately from engine time.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from typing import Protocol

from indicators import DESCRIPTIONS
from regime import INDICATORS

log = logging.getLogger(__name__)

# Measured on the strategy prompt (Oct 2026): gemini-3.8-flash takes ~35 s by
# default because it "thinks" first (~850 tokens) and streams slowly; ~11 s with
# thinking disabled. gemini-3.5-flash-lite takes ~3.5 s and keeps the citation
# discipline, so it is the default. Override with GEMINI_MODEL.
DEFAULT_MODEL = "gemini-3.5-flash-lite"
TOP_CONDITIONS = 5

# What the model must return. Kept small so it is easy to validate and render.
EXPLANATION_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string", "description": "Two or three sentences for a student: the main point."},
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "heading": {"type": "string"},
                    "body": {"type": "string", "description": "One to three short paragraphs. Plain language."},
                    "basis": {"type": "string", "enum": ["fact", "inference"],
                              "description": "'fact' if every claim restates a provided fact; 'inference' if it offers an interpretation or mechanism."},
                    "evidence": {"type": "array", "items": {"type": "string"},
                                 "description": "Fact keys (from the provided facts) this section relies on."},
                },
                "required": ["heading", "body", "basis", "evidence"],
            },
        },
        "caveats": {"type": "array", "items": {"type": "string"},
                    "description": "Limits of what the data can show. At least one."},
    },
    "required": ["summary", "sections", "caveats"],
}

SYSTEM_PROMPT = """You are a patient quantitative-finance teacher writing for someone learning about systematic trading.

You will be given FACTS computed by a backtesting engine, as a JSON object whose keys are fact ids, plus reference definitions of market indicators. Rules:
1. Use only the facts and definitions provided. Do not introduce outside numbers, dates, or events. If you refer to a number, quote it exactly as given.
2. Every section must list, in "evidence", the fact ids it relies on. If a section explains a mechanism or offers an interpretation rather than restating facts, set "basis" to "inference" and say so in the text ("one plausible reason is...").
3. Explain the economic mechanism: why would this strategy do well or badly under these conditions? Connect each indicator you mention to the strategy's logic.
4. Be honest about weak evidence: small effects, few trades, short periods, or alpha below the significance bar.
5. No investment advice, no predictions about the future, no hype. Write in plain language and define any jargon the first time you use it.
6. Keep it compact: 3 to 5 sections, each one to three short paragraphs."""

STRATEGY_TASK = """Explain this strategy's backtest to a learner. Cover: (a) what the strategy does, (b) which market indicators mattered most and why those make sense for this strategy's logic, (c) the kind of market it needs, (d) the kind of market in which it fails and the mechanism of failure, and (e) how to read the alpha result, including the before/after-publication comparison if present."""

PERIOD_TASK = """Explain how this strategy performed during the selected period, for a learner. Cover: (a) what happened to the strategy versus its benchmark in this window, (b) what the market looked like during the window compared with the rest of history, (c) why the strategy's logic would produce this outcome under those conditions, and (d) what general lesson about this strategy type the episode illustrates."""


# ---------------------------------------------------------------------------
# Facts
# ---------------------------------------------------------------------------

def _pct(x: float) -> str:
    return f"{x * 100:+.1f}%"


def _ordinal(n: float) -> str:
    n = int(round(n))
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _fmt_value(x: float, fmt: str) -> str:
    if fmt == "pct":
        return f"{x * 100:.1f}%"
    if fmt == "level":
        return f"{x:+.2f}"
    return f"{x:.2f}"


def _condition_facts(prefix: str, label: str, conditions: list[dict], facts: dict) -> None:
    """Top conditions for one regime, as individual facts."""
    for rank, c in enumerate(conditions[:TOP_CONDITIONS], 1):
        facts[f"{prefix}_{rank}"] = (
            f"{label}: {c['label']} was {c['direction']} ({c['band']}; effect {c['effect']:+.2f}). "
            f"Median {_fmt_value(c['inside_median'], c['fmt'])} on those days vs "
            f"{_fmt_value(c['baseline_median'], c['fmt'])} on other days; "
            f"that median sits at the {_ordinal(c['percentile'])} percentile of all history. Indicator key: {c['indicator']}."
        )


def _alpha_facts(alpha: dict, facts: dict) -> None:
    def line(name: str, a: dict | None, period: str) -> None:
        if a is None:
            facts[name] = f"Alpha {period}: not measured (fewer than 2 years of data)."
            return
        strength = ("statistically significant (clears the t >= 3 bar for a result picked from many strategies)"
                    if abs(a["t_stat"]) >= 3 else
                    "suggestive but below the t >= 3 bar" if abs(a["t_stat"]) >= 2 else
                    "indistinguishable from zero")
        facts[name] = (f"Alpha {period} ({a['start']} to {a['end']}, {a['years']} years): {_pct(a['alpha'])} per year, "
                       f"t-stat {a['t_stat']:.2f} ({strength}), beta {a['beta']:.2f}.")

    line("alpha_full", alpha.get("full"), "over the full sample")
    pub = alpha.get("published")
    if pub:
        facts["publication_year"] = f"The strategy was published in {pub}."
        line("alpha_before_publication", alpha.get("before"), f"before publication ({pub})")
        line("alpha_after_publication", alpha.get("after"), f"after publication ({pub})")


def _metric_facts(metrics: dict, benchmark: str, facts: dict) -> None:
    s, b = metrics["strategy"], metrics["benchmark"]
    facts["cagr"] = f"Compound annual growth: strategy {_pct(s['cagr'])}, {benchmark} {_pct(b['cagr'])}."
    facts["volatility"] = f"Annualized volatility: strategy {s['volatility'] * 100:.1f}%, {benchmark} {b['volatility'] * 100:.1f}%."
    facts["sharpe"] = f"Sharpe ratio (return above the Fed funds rate per unit of volatility): strategy {s['sharpe']:.2f}, {benchmark} {b['sharpe']:.2f}."
    facts["max_drawdown"] = f"Worst peak-to-trough loss: strategy {_pct(s['max_drawdown'])}, {benchmark} {_pct(b['max_drawdown'])}."


def strategy_facts(result: dict, spec: dict, params: dict) -> dict:
    """Facts for 'explain this strategy'. Contains no prices, dates series or trades."""
    bench = "cash" if result["benchmark"] == "CASH" else result["benchmark"]
    facts = {
        "strategy": f"{spec['name']}: {spec['description']}",
        "parameters": "Parameters: " + ", ".join(f"{k}={v}" for k, v in params.items()) + ".",
        "period": f"Backtest period: {result['dates'][0]} to {result['dates'][-1]} ({len(result['dates']):,} trading days). Benchmark: {bench}.",
    }
    if spec.get("source"):
        facts["source"] = f"Source: {spec['source']}."
    for k, v in result.get("diagnostics", {}).items():
        facts[f"diag_{k.lower().replace(' ', '_')}"] = f"{k}: {v}."
    _metric_facts(result["metrics"], bench, facts)
    _alpha_facts(result["alpha"], facts)
    days = result.get("regime_days", {})
    if days:
        facts["regime_days"] = ("Days by regime (rolling 3-month return vs benchmark, terciles): "
                                + ", ".join(f"{k} {v:,}" for k, v in days.items()) + ".")
    _condition_facts("outperform", "On days the strategy outperformed", result["conditions"]["outperform"], facts)
    _condition_facts("underperform", "On days it underperformed", result["conditions"]["underperform"], facts)
    return facts


def period_facts(stats: dict, spec: dict, params: dict, conditions: list[dict]) -> dict:
    """Facts for 'explain this period'. `stats` comes from backtest.period_stats."""
    bench = "cash" if stats["benchmark"] == "CASH" else stats["benchmark"]
    facts = {
        "strategy": f"{spec['name']}: {spec['description']}",
        "parameters": "Parameters: " + ", ".join(f"{k}={v}" for k, v in params.items()) + ".",
        "window": f"Selected period: {stats['start']} to {stats['end']} ({stats['n_days']} trading days). Benchmark: {bench}.",
        "window_return": f"Over the period the strategy returned {_pct(stats['strategy_return'])} and {bench} returned {_pct(stats['benchmark_return'])}: a difference of {_pct(stats['excess_return'])}.",
        "window_drawdown": f"Worst peak-to-trough loss inside the period: strategy {_pct(stats['strategy_max_drawdown'])}, {bench} {_pct(stats['benchmark_max_drawdown'])}.",
        "window_volatility": f"Annualized volatility inside the period: strategy {stats['strategy_volatility'] * 100:.1f}%, {bench} {stats['benchmark_volatility'] * 100:.1f}%.",
        "exposure": f"Average gross exposure during the period: {stats['avg_gross_exposure']:.2f}x (1.0 = fully invested); the strategy held a position on {stats['days_invested_pct']:.0f}% of days and changed its position {stats['position_changes']} times.",
        "best_worst_day": f"Strategy's best single day {_pct(stats['best_day'])} and worst {_pct(stats['worst_day'])} within the period.",
    }
    if stats.get("regime_share"):
        facts["regime_share"] = ("Share of the period's days labelled by the regime segmenter: "
                                 + ", ".join(f"{k} {v:.0f}%" for k, v in stats["regime_share"].items()) + ".")
    for rank, c in enumerate(conditions[:TOP_CONDITIONS], 1):
        facts[f"condition_{rank}"] = (
            f"During the period, {c['label']} was {c['direction']} ({c['band']}; effect {c['effect']:+.2f}): "
            f"median {_fmt_value(c['inside_median'], c['fmt'])} vs {_fmt_value(c['baseline_median'], c['fmt'])} "
            f"for the rest of history ({_ordinal(c['percentile'])} percentile). Indicator key: {c['indicator']}."
        )
    return facts


def indicator_definitions(facts: dict) -> dict:
    """Reference text for every indicator mentioned in the facts."""
    text = " ".join(facts.values())
    return {
        key: {"label": INDICATORS[key]["label"], **DESCRIPTIONS[key]}
        for key in INDICATORS if f"Indicator key: {key}." in text
    }


# ---------------------------------------------------------------------------
# Explainers
# ---------------------------------------------------------------------------

class Explainer(Protocol):
    name: str

    def explain(self, task: str, facts: dict, definitions: dict) -> dict: ...


class TemplateExplainer:
    """Deterministic explanation assembled from the facts. No model, no network."""

    name = "template"

    def explain(self, task: str, facts: dict, definitions: dict) -> dict:
        is_period = "window" in facts
        sections = []
        if is_period:
            sections.append({"heading": "What happened", "basis": "fact",
                             "evidence": ["window", "window_return", "window_drawdown", "exposure"],
                             "body": " ".join(facts[k] for k in ("window_return", "window_drawdown", "exposure"))})
            cond_keys = [k for k in facts if k.startswith("condition_")]
            if cond_keys:
                sections.append({"heading": "What the market looked like", "basis": "fact", "evidence": cond_keys,
                                 "body": " ".join(facts[k] for k in cond_keys)})
        else:
            sections.append({"heading": "What the strategy does", "basis": "fact",
                             "evidence": ["strategy", "parameters", "period"],
                             "body": f"{facts['strategy']} {facts['parameters']} {facts['period']}"})
            sections.append({"heading": "Performance", "basis": "fact",
                             "evidence": ["cagr", "volatility", "sharpe", "max_drawdown"],
                             "body": " ".join(facts[k] for k in ("cagr", "volatility", "sharpe", "max_drawdown"))})
            out = [k for k in facts if k.startswith("outperform_")]
            under = [k for k in facts if k.startswith("underperform_")]
            if out:
                sections.append({"heading": "When it worked", "basis": "fact", "evidence": out,
                                 "body": " ".join(facts[k] for k in out)})
            if under:
                sections.append({"heading": "When it failed", "basis": "fact", "evidence": under,
                                 "body": " ".join(facts[k] for k in under)})
            alpha_keys = [k for k in facts if k.startswith("alpha_") or k == "publication_year"]
            sections.append({"heading": "Alpha", "basis": "fact", "evidence": alpha_keys,
                             "body": " ".join(facts[k] for k in alpha_keys)})
        if definitions:
            sections.append({"heading": "Indicator definitions", "basis": "reference", "evidence": [],
                             "body": " ".join(f"{d['label']}: {d['what']} {d['matters']}" for d in definitions.values())})
        summary = (facts["window_return"] if is_period
                   else f"{facts['cagr']} {facts['max_drawdown']} {facts['alpha_full']}")
        return {
            "summary": summary + " (Offline template: these sentences restate the engine's facts; no AI interpretation.)",
            "sections": sections,
            "caveats": ["No AI model was used for this explanation, so it contains no interpretation of mechanisms.",
                        "Backtest results assume trading at the closing price with no transaction costs."],
        }


class GeminiExplainer:
    """Calls Gemini with a JSON schema; falls back to the template on any failure."""

    name = "gemini"

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        from google import genai
        from google.genai import types
        self._types = types
        self._client = genai.Client(api_key=api_key)
        self.model = model
        self._no_thinking_unsupported = False

    def explain(self, task: str, facts: dict, definitions: dict) -> dict:
        prompt = (f"{task}\n\nFACTS (json, keyed by fact id):\n{json.dumps(facts, indent=1)}\n\n"
                  f"INDICATOR DEFINITIONS (json):\n{json.dumps(definitions, indent=1)}")
        base = dict(system_instruction=SYSTEM_PROMPT, response_mime_type="application/json",
                    response_json_schema=EXPLANATION_SCHEMA, temperature=0.2)
        # Extended thinking triples latency on this task and adds nothing the
        # schema doesn't already force, so turn it off where the model allows.
        # Models that reject the option (400) are retried without it, and the
        # answer is remembered for the process lifetime.
        config = self._types.GenerateContentConfig(
            **base, **({} if self._no_thinking_unsupported else
                       {"thinking_config": self._types.ThinkingConfig(thinking_budget=0)}))
        for attempt in range(3):
            try:
                response = self._client.models.generate_content(model=self.model, contents=prompt, config=config)
                return json.loads(response.text)
            except Exception as e:
                code = getattr(e, "code", None) or getattr(e, "status_code", None)
                if code == 400 and not self._no_thinking_unsupported:
                    self._no_thinking_unsupported = True
                    config = self._types.GenerateContentConfig(**base)
                    continue
                if attempt < 2 and code in (429, 500, 503):  # transient overload: retry once
                    log.warning("Gemini returned %s; retrying once", code)
                    time.sleep(2)
                    continue
                raise


def validate(explanation: dict, facts: dict) -> dict:
    """Drop citations to facts that don't exist and flag the sections that had them."""
    valid = set(facts)
    bad_total = 0
    for section in explanation.get("sections", []):
        cited = section.get("evidence", []) or []
        good = [k for k in cited if k in valid]
        bad = len(cited) - len(good)
        bad_total += bad
        section["evidence"] = good
        section["unsupported_citations"] = bad
        if not good and section.get("basis") == "fact":
            # A "fact" section with no verifiable evidence is downgraded.
            section["basis"] = "inference"
        if section.get("basis") not in ("fact", "inference", "reference"):
            section["basis"] = "inference"
    explanation["unsupported_citations"] = bad_total
    return explanation


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class ExplanationService:
    """Chooses the explainer, caches by facts, and times the model separately."""

    def __init__(self, explainer: Explainer | None = None):
        self.explainer = explainer
        self.fallback = TemplateExplainer()
        self._cache: dict[str, dict] = {}

    @classmethod
    def from_env(cls) -> "ExplanationService":
        key = os.environ.get("GEMINI_API_KEY")
        model = os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)
        if not key:
            log.warning("GEMINI_API_KEY not set; explanations will use the offline template")
            return cls(None)
        try:
            return cls(GeminiExplainer(key, model))
        except Exception as e:  # SDK missing or misconfigured
            log.warning("Gemini unavailable (%s); explanations will use the offline template", e)
            return cls(None)

    @property
    def provider(self) -> str:
        return self.explainer.name if self.explainer else self.fallback.name

    def explain(self, task: str, facts: dict) -> dict:
        definitions = indicator_definitions(facts)
        cache_key = hashlib.sha256(json.dumps([self.provider, task, facts], sort_keys=True).encode()).hexdigest()
        if cache_key in self._cache:
            return {**self._cache[cache_key], "cached": True}

        provider, error = self.provider, None
        start = time.perf_counter()
        try:
            if self.explainer is None:
                raise RuntimeError("no model configured")
            explanation = self.explainer.explain(task, facts, definitions)
        except Exception as e:
            if self.explainer is not None:
                log.warning("explainer %s failed (%s); using template", self.explainer.name, e)
                error = f"{type(e).__name__}: {e}"[:200]
            provider = self.fallback.name
            explanation = self.fallback.explain(task, facts, definitions)
        model_ms = (time.perf_counter() - start) * 1000

        result = {
            **validate(explanation, facts),
            "facts": facts,
            "definitions": definitions,
            "provider": provider,
            "model": getattr(self.explainer, "model", None) if provider == "gemini" else None,
            "model_latency_ms": round(model_ms, 1),
            "fallback_reason": error,
            "cached": False,
        }
        if error is None:  # a fallback must not pin the template until restart
            self._cache[cache_key] = result
        return result
