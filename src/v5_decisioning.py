from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline

from .modeling import FEATURES, NUMERIC, CATEGORICAL, ks_stat, preprocessing
from .scorecard import WOELogisticScorecard

ROOT = Path(__file__).resolve().parents[1]
BUILD = "V5.0"
MODEL_ID = "RCS_PD_4.0"
POLICY_VERSION = "RETAIL_POLICY_2.0_CANDIDATE"

LGD = 0.65
CCF = 1.00
FUNDING_RATE = 0.07
OPERATING_COST = 350_000
CAPITAL_RATE = 0.10


def _metric_row(name: str, y: pd.Series, pd12: np.ndarray) -> dict:
    auc = float(roc_auc_score(y, pd12))
    return {
        "window": name,
        "applications": int(len(y)),
        "bad_rate": float(np.mean(y)),
        "mean_pd": float(np.mean(pd12)),
        "auc": auc,
        "gini": 2 * auc - 1,
        "ks": ks_stat(y, pd12),
        "brier": float(brier_score_loss(y, pd12)),
        "calibration_gap": float(np.mean(pd12) - np.mean(y)),
    }


def _challenger() -> Pipeline:
    return Pipeline(
        [
            ("prep", preprocessing(True)),
            (
                "model",
                HistGradientBoostingClassifier(
                    max_iter=180,
                    max_leaf_nodes=15,
                    learning_rate=0.055,
                    min_samples_leaf=45,
                    l2_regularization=2.0,
                    random_state=42,
                ),
            ),
        ]
    )


def walk_forward(data: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    boundaries = [0.45, 0.58, 0.71, 0.84, 1.00]
    champion_rows, comparison_rows = [], []
    for fold in range(4):
        train_end = int(boundaries[fold] * len(data))
        test_end = int(boundaries[fold + 1] * len(data))
        train, test = data.iloc[:train_end], data.iloc[train_end:test_end]
        champion = WOELogisticScorecard().fit(train[FEATURES], train.default_12m)
        challenger = _challenger().fit(train[FEATURES], train.default_12m)
        for model_name, model in [("WOE logistic champion", champion), ("Gradient boosting challenger", challenger)]:
            prob = model.predict_proba(test[FEATURES])[:, 1]
            row = _metric_row(f"Fold {fold + 1}", test.default_12m, prob)
            row.update(
                {
                    "model": model_name,
                    "train_end": str(train.application_date.max()),
                    "test_start": str(test.application_date.min()),
                    "test_end": str(test.application_date.max()),
                }
            )
            comparison_rows.append(row)
            if model_name.startswith("WOE"):
                champion_rows.append(row)
    return champion_rows, comparison_rows


def economics(frame: pd.DataFrame, pd_col: str = "pd_12m", lgd: float = LGD) -> pd.DataFrame:
    x = frame.copy()
    x["ead"] = np.minimum(x.requested_amount, x.monthly_income * 5.0) * CCF
    x["interest_rate"] = np.select(
        [x[pd_col] < 0.05, x[pd_col] < 0.10, x[pd_col] < 0.18],
        [0.16, 0.18, 0.22],
        default=0.25,
    )
    x["term_years"] = x.tenor_months / 12
    x["expected_loss"] = x[pd_col] * lgd * x.ead
    x["interest_income"] = x.interest_rate * x.ead * x.term_years
    x["funding_cost"] = FUNDING_RATE * x.ead * x.term_years
    x["operating_cost"] = OPERATING_COST
    x["expected_profit"] = x.interest_income - x.expected_loss - x.funding_cost - x.operating_cost
    x["economic_capital"] = np.maximum(CAPITAL_RATE * x.ead, 1)
    x["raroc"] = x.expected_profit / x.economic_capital / np.maximum(x.term_years, 0.5)
    x["break_even_rate"] = FUNDING_RATE + (x.expected_loss + x.operating_cost) / np.maximum(x.ead * x.term_years, 1)
    return x


def strategy_grid(oot: pd.DataFrame) -> list[dict]:
    rows = []
    for approve_max in np.arange(0.06, 0.131, 0.01):
        for review_max in np.arange(max(approve_max + 0.04, 0.14), 0.231, 0.02):
            x = economics(oot)
            x["decision"] = np.where(x.pd_12m < approve_max, "APPROVE", np.where(x.pd_12m < review_max, "REVIEW", "DECLINE"))
            booked = x[x.decision == "APPROVE"]
            rows.append(
                {
                    "approve_pd_max": round(float(approve_max), 4),
                    "review_pd_max": round(float(review_max), 4),
                    "approval_rate": float(len(booked) / len(x)),
                    "review_rate": float((x.decision == "REVIEW").mean()),
                    "observed_bad_rate": float(booked.default_12m.mean()) if len(booked) else None,
                    "mean_pd": float(booked.pd_12m.mean()) if len(booked) else None,
                    "total_ead": float(booked.ead.sum()),
                    "expected_loss": float(booked.expected_loss.sum()),
                    "expected_profit": float(booked.expected_profit.sum()),
                    "portfolio_raroc": float(booked.expected_profit.sum() / max(booked.economic_capital.sum(), 1)),
                }
            )
    return rows


def stress_test(oot: pd.DataFrame, approve_max: float = 0.10) -> list[dict]:
    scenarios = [
        ("Base", 1.00, LGD, 0.00),
        ("Downturn", 1.35, 0.72, 0.015),
        ("Severe", 1.75, 0.80, 0.035),
    ]
    rows = []
    base_logit = np.log(np.clip(oot.pd_12m, 1e-6, 1 - 1e-6) / np.clip(1 - oot.pd_12m, 1e-6, 1))
    for scenario, odds_multiplier, lgd, funding_shock in scenarios:
        stressed_pd = 1 / (1 + np.exp(-(base_logit + np.log(odds_multiplier))))
        x = oot.assign(stressed_pd=stressed_pd)
        x = economics(x, "stressed_pd", lgd)
        x["funding_cost"] = (FUNDING_RATE + funding_shock) * x.ead * x.term_years
        x["expected_profit"] = x.interest_income - x.expected_loss - x.funding_cost - x.operating_cost
        booked = x[x.stressed_pd < approve_max]
        rows.append(
            {
                "scenario": scenario,
                "pd_odds_multiplier": odds_multiplier,
                "lgd": lgd,
                "funding_rate": FUNDING_RATE + funding_shock,
                "approval_rate": float(len(booked) / len(x)),
                "portfolio_pd": float(booked.stressed_pd.mean()) if len(booked) else None,
                "expected_loss": float(booked.expected_loss.sum()),
                "expected_profit": float(booked.expected_profit.sum()),
                "portfolio_raroc": float(booked.expected_profit.sum() / max(booked.economic_capital.sum(), 1)),
            }
        )
    return rows


def build() -> dict:
    data = pd.read_csv(ROOT / "data/development_sample.csv", parse_dates=["application_date"]).sort_values("application_date").reset_index(drop=True)
    model = joblib.load(ROOT / "artifacts/retail_woe_scorecard_v4.joblib")
    split = int(0.80 * len(data))
    oot = data.iloc[split:].copy()
    oot["pd_12m"] = model.predict_proba(oot[FEATURES])[:, 1]

    wf_champion, wf_comparison = walk_forward(data)
    strategies = strategy_grid(oot)
    stresses = stress_test(oot)
    priced = economics(oot)
    priced["risk_band"] = pd.cut(priced.pd_12m, [0, .05, .10, .18, 1], labels=["A", "B", "C", "D"], include_lowest=True)
    profitability = priced.groupby("risk_band", observed=True).agg(
        applications=("application_id", "size"), mean_pd=("pd_12m", "mean"), total_ead=("ead", "sum"),
        expected_loss=("expected_loss", "sum"), expected_profit=("expected_profit", "sum"),
        mean_raroc=("raroc", "mean"), mean_break_even_rate=("break_even_rate", "mean")
    ).reset_index()

    best = max((r for r in strategies if r["observed_bad_rate"] is not None and r["observed_bad_rate"] <= 0.10), key=lambda r: r["expected_profit"])
    champion_avg_auc = float(np.mean([r["auc"] for r in wf_comparison if r["model"].startswith("WOE")]))
    challenger_avg_auc = float(np.mean([r["auc"] for r in wf_comparison if r["model"].startswith("Gradient")]))
    challenger_decision = "RETAIN CHAMPION" if challenger_avg_auc - champion_avg_auc < 0.02 else "VALIDATE CHALLENGER"
    summary = {
        "build": BUILD,
        "model_id": MODEL_ID,
        "candidate_policy_version": POLICY_VERSION,
        "calculated_at": datetime.now(timezone.utc).isoformat(),
        "walk_forward_folds": 4,
        "champion_mean_auc": champion_avg_auc,
        "challenger_mean_auc": challenger_avg_auc,
        "challenger_auc_uplift": challenger_avg_auc - champion_avg_auc,
        "challenger_decision": challenger_decision,
        "recommended_approve_pd_max": best["approve_pd_max"],
        "recommended_review_pd_max": best["review_pd_max"],
        "recommended_approval_rate": best["approval_rate"],
        "recommended_bad_rate": best["observed_bad_rate"],
        "recommended_expected_profit": best["expected_profit"],
        "recommended_raroc": best["portfolio_raroc"],
        "status": "PASS" if min(r["auc"] for r in wf_champion) >= 0.65 and stresses[-1]["expected_profit"] > 0 else "REVIEW",
    }

    payload_path = ROOT / "reports/validation_payload.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload.update(
        {
            "v5_summary": summary,
            "v5_walk_forward": wf_champion,
            "v5_champion_challenger": wf_comparison,
            "v5_strategy_grid": strategies,
            "v5_stress_testing": stresses,
            "v5_profitability": profitability.astype(object).where(pd.notnull(profitability), None).to_dict("records"),
        }
    )
    payload_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    (ROOT / "artifacts/model_metadata_v5.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(strategies).to_csv(ROOT / "data/decision_strategy_v5.csv", index=False)
    pd.DataFrame(stresses).to_csv(ROOT / "data/stress_scenarios_v5.csv", index=False)
    return summary


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
