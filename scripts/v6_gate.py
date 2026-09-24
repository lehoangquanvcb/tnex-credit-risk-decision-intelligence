import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
p = json.loads((ROOT / "reports/validation_payload.json").read_text())
checks = {
    "LGD validation": p["v6_lgd_validation"]["status"] == "PASS",
    "EAD validation": p["v6_ead_validation"]["status"] == "PASS",
    "Scenario weights": abs(sum(x["weight"] for x in p["v6_macro_scenarios"]) - 1) < 1e-9,
    "ECL reconciliation": abs(sum(x["weighted_ecl"] for x in p["v6_ifrs9_stage_summary"]) - p["v6_summary"]["weighted_ecl"]) < 1,
    "Unsafe rollout blocked": p["v6_deployment_gates"][2]["status"] == "ROLLBACK" and p["v6_deployment_gates"][3]["status"] == "BLOCKED",
    "Rollback evidence": all(x["status"] == "PASS" for x in p["v6_rollback_drill"]),
}
print(checks)
if not all(checks.values()):
    raise SystemExit("V6 bank-grade approval gate failed")
