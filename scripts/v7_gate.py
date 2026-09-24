import json
from pathlib import Path

p = json.loads((Path(__file__).resolve().parents[1] / "reports/validation_payload.json").read_text())
checks = {
    "three configured products": len(p["v7_product_config"]) == 3,
    "real-time p95 under five seconds": p["v7_summary"]["p95_end_to_end_ms"] < 5000,
    "fraud and credit decisions present": all("fraud_score" in x and "decision_pd" in x for x in p["v7_fraud_credit"]),
    "funnel reconciles": all(a["customers"] >= b["customers"] for a,b in zip(p["v7_funnel"], p["v7_funnel"][1:])),
    "collections reminder suppression passes": next(x for x in p["v7_collections"] if x["control"] == "Reminder suppression")["exceptions"] == 0,
}
for name, ok in checks.items(): print(f'{"PASS" if ok else "FAIL"}: {name}')
if not all(checks.values()): raise SystemExit(1)
