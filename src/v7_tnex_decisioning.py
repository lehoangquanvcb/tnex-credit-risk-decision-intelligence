from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .modeling import FEATURES

ROOT = Path(__file__).resolve().parents[1]
BUILD = "V7.0"
MODEL_ID = "RCS_PD_4.0"

PRODUCTS = {
    "CASH_LOAN": {"label": "Cash Loan", "max_limit_vnd": 70_000_000, "approve_pd_max": .12, "review_pd_max": .20, "fraud_max": .35},
    "BNPL": {"label": "Buy Now Pay Later", "max_limit_vnd": 25_000_000, "approve_pd_max": .10, "review_pd_max": .17, "fraud_max": .30},
    "BUSINESS_LOAN": {"label": "Business Loan", "max_limit_vnd": 100_000_000, "approve_pd_max": .14, "review_pd_max": .22, "fraud_max": .35},
}


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def build() -> dict:
    rng = np.random.default_rng(707)
    data = pd.read_csv(ROOT / "data/development_sample.csv", parse_dates=["application_date"]).tail(2400).copy().reset_index(drop=True)
    model = joblib.load(ROOT / "artifacts/retail_woe_scorecard_v4.joblib")
    data["base_pd"] = model.predict_proba(data[FEATURES])[:, 1]
    data["product"] = rng.choice(list(PRODUCTS), len(data), p=[.55, .30, .15])
    data["identity_match_score"] = np.clip(rng.beta(18, 2, len(data)), 0, 1)
    data["device_age_days"] = np.maximum(0, rng.gamma(2.2, 120, len(data))).round()
    data["application_velocity_24h"] = rng.poisson(.55, len(data))
    data["failed_txn_rate"] = np.clip(rng.beta(1.3, 18, len(data)), 0, 1)
    data["avg_balance_90d"] = np.maximum(0, data.monthly_income * rng.uniform(.08, .75, len(data)))
    data["transaction_count_90d"] = rng.poisson(45, len(data))
    data["income_regularly"] = rng.binomial(1, np.where(data.employment_type == "Salaried", .82, .54))
    data["sales_monthly"] = np.where(data["product"] == "BUSINESS_LOAN", data.monthly_income * rng.uniform(2.5, 7, len(data)), 0)
    data["prior_loans"] = rng.poisson(.8, len(data))
    data["prior_on_time_ratio"] = np.where(data.prior_loans > 0, np.clip(rng.beta(12, 2, len(data)), 0, 1), np.nan)

    fraud_logit = -4.1 + 4.2 * (1 - data.identity_match_score) + .8 * (data.device_age_days < 7) + .65 * data.application_velocity_24h + 3.0 * data.failed_txn_rate
    data["fraud_score"] = 1 / (1 + np.exp(-fraud_logit))
    behaviour_adj = -.22 * data.income_regularly - .002 * np.minimum(data.transaction_count_90d, 100) - .30 * data.prior_on_time_ratio.fillna(.75) + .75 * data.failed_txn_rate
    data["decision_pd"] = 1 / (1 + np.exp(-(_logit(data.base_pd) + behaviour_adj)))

    decisions, limits, reasons = [], [], []
    for r in data.itertuples():
        cfg = PRODUCTS[r.product]
        affordability = max(0, r.monthly_income * (1 - r.existing_dti) * 3.2)
        business_capacity = r.sales_monthly * .55 if r.product == "BUSINESS_LOAN" else affordability
        limit = min(r.requested_amount, cfg["max_limit_vnd"], business_capacity)
        if r.fraud_score > cfg["fraud_max"] or r.identity_match_score < .55:
            decision, reason = "DECLINE", "FRAUD_RISK"
        elif r.fraud_score > cfg["fraud_max"] * .72 or r.decision_pd >= cfg["review_pd_max"]:
            decision, reason = "REVIEW", "ENHANCED_DUE_DILIGENCE"
        elif r.decision_pd < cfg["approve_pd_max"] and limit >= 1_000_000:
            decision, reason = "APPROVE", "POLICY_PASS"
        else:
            decision, reason = "REVIEW", "CREDIT_OR_AFFORDABILITY"
        decisions.append(decision); limits.append(float(max(0, limit))); reasons.append(reason)
    data["decision"], data["approved_limit_vnd"], data["primary_reason"] = decisions, limits, reasons

    data["ekyc_ms"] = np.clip(rng.normal(620, 150, len(data)), 180, 1400)
    data["feature_ms"] = np.clip(rng.normal(310, 85, len(data)), 80, 900)
    data["fraud_ms"] = np.clip(rng.normal(245, 65, len(data)), 60, 700)
    data["credit_ms"] = np.clip(rng.normal(32, 9, len(data)), 8, 90)
    data["policy_offer_ms"] = np.clip(rng.normal(45, 12, len(data)), 12, 130)
    data["end_to_end_ms"] = data[["ekyc_ms", "feature_ms", "fraud_ms", "credit_ms", "policy_offer_ms"]].sum(axis=1)

    product_rows = []
    for product, f in data.groupby("product"):
        product_rows.append({"product": product, "applications": int(len(f)), "approval_rate": float((f.decision == "APPROVE").mean()),
            "review_rate": float((f.decision == "REVIEW").mean()), "fraud_decline_rate": float((f.primary_reason == "FRAUD_RISK").mean()),
            "mean_decision_pd": float(f.decision_pd.mean()), "approved_limit_vnd": float(f.loc[f.decision == "APPROVE", "approved_limit_vnd"].sum()),
            "p95_latency_ms": float(f.end_to_end_ms.quantile(.95))})

    started = len(data); identity = int(started * .91); submitted = int(identity * .90); decisioned = submitted; approved = int((data.decision == "APPROVE").sum() * submitted / started); disbursed = int(approved * .84)
    funnel_counts = [("Application started", started), ("Identity verified", identity), ("Application submitted", submitted), ("Decision rendered", decisioned), ("Approved", approved), ("Disbursed / activated", disbursed)]
    funnel = [{"stage": s, "customers": n, "conversion_from_prior": 1.0 if i == 0 else n / funnel_counts[i-1][1], "conversion_from_start": n / started} for i, (s, n) in enumerate(funnel_counts)]

    features = [
        ("Bureau", "bureau_score", "At decision", "No", "Refer / conservative policy", "Credit Risk"),
        ("Transaction", "avg_balance_90d", "Daily", "Yes", "Income-only model", "Data Office"),
        ("Transaction", "transaction_count_90d", "Daily", "Yes", "Bureau-only model", "Data Office"),
        ("Device", "device_age_days", "Real time", "Yes", "Step-up verification", "Fraud Risk"),
        ("Identity", "identity_match_score", "Real time", "Yes", "Manual identity review", "Financial Crime"),
        ("Business", "sales_monthly", "Daily", "Yes", "Manual bank-statement review", "Business Credit"),
        ("Lifecycle", "prior_on_time_ratio", "Daily", "Yes", "No repeat-loan uplift", "Collections"),
    ]
    feature_rows = [{"feature_group": a, "feature": b, "freshness_sla": c, "consent_required": d, "fallback": e, "owner": f} for a,b,c,d,e,f in features]

    eligible = data[data.prior_loans > 0].copy()
    eligible["lifecycle_action"] = np.select([
        (eligible.prior_on_time_ratio >= .95) & (eligible.decision_pd < .08),
        (eligible.prior_on_time_ratio < .70) | (eligible.decision_pd >= .20),
        (eligible.prior_on_time_ratio < .85)], ["INCREASE_LIMIT", "SUPPRESS_OFFER", "REDUCE_LIMIT"], default="MAINTAIN")
    lifecycle = eligible.groupby("lifecycle_action").agg(customers=("application_id", "size"), mean_pd=("decision_pd", "mean"), mean_on_time_ratio=("prior_on_time_ratio", "mean")).reset_index().to_dict("records")
    collections = [
        {"control": "Payment posting reconciliation", "population": 1840, "exceptions": 21, "rate": 21/1840, "sla": "T+1", "action": "Suspend reminders until ledger match", "status": "PASS"},
        {"control": "Reminder suppression", "population": 21, "exceptions": 0, "rate": 0.0, "sla": "Immediate", "action": "No collection message on unresolved payment", "status": "PASS"},
        {"control": "Hardship / complaint routing", "population": 63, "exceptions": 2, "rate": 2/63, "sla": "4 hours", "action": "Specialist review and vulnerability flag", "status": "AMBER"},
        {"control": "Promise-to-pay monitoring", "population": 214, "exceptions": 8, "rate": 8/214, "sla": "Daily", "action": "Re-segment and next-best action", "status": "PASS"},
    ]
    latency = [{"component": c.replace("_ms", "").replace("_", " ").title(), "p50_ms": float(data[c].median()), "p95_ms": float(data[c].quantile(.95)), "sla_ms": 5000 if c == "end_to_end_ms" else None} for c in ["ekyc_ms","feature_ms","fraud_ms","credit_ms","policy_offer_ms","end_to_end_ms"]]
    configs = [{"product": k, **v, "policy_version": "TNEX_SIM_1.0", "effective_date": "2026-09-24"} for k,v in PRODUCTS.items()]
    summary = {"build": BUILD, "portfolio": "TNEX-aligned digital lending simulation", "model_id": MODEL_ID,
        "calculated_at": datetime.now(timezone.utc).isoformat(), "applications": started, "products": len(PRODUCTS),
        "approval_rate": float((data.decision == "APPROVE").mean()), "fraud_decline_rate": float((data.primary_reason == "FRAUD_RISK").mean()),
        "p95_end_to_end_ms": float(data.end_to_end_ms.quantile(.95)), "disbursement_conversion": disbursed / started,
        "overall_status": "PASS — SIMULATION", "disclaimer": "Portfolio simulation only; not an official TNEX model or affiliated implementation."}
    payload_path = ROOT / "reports/validation_payload.json"
    payload = json.loads(payload_path.read_text())
    payload.update({"v7_summary": summary, "v7_product_config": configs, "v7_product_performance": product_rows,
        "v7_fraud_credit": data[["application_id","product","base_pd","decision_pd","fraud_score","decision","approved_limit_vnd","primary_reason","end_to_end_ms"]].head(500).to_dict("records"),
        "v7_latency": latency, "v7_funnel": funnel, "v7_feature_governance": feature_rows, "v7_lifecycle": lifecycle, "v7_collections": collections})
    payload_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    (ROOT / "artifacts/model_metadata_v7.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    data.to_csv(ROOT / "data/tnex_decisions_v7.csv", index=False)
    pd.DataFrame(funnel).to_csv(ROOT / "data/digital_funnel_v7.csv", index=False)
    pd.DataFrame(collections).to_csv(ROOT / "data/collections_controls_v7.csv", index=False)
    return summary


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
