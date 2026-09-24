import json
from pathlib import Path
p=json.loads((Path(__file__).resolve().parents[1]/"reports/validation_payload.json").read_text())
checks={
 "three product models":len(p["v8_product_models"])==3,
 "strategy is profitable":p["v8_recommended_strategy"]["contribution_vnd"]>0,
 "strategy satisfies minimum approval":p["v8_recommended_strategy"]["approval_rate"]>=.55,
 "fraud graph populated":len(p["v8_fraud_network"])>0,
 "payment exception suppresses contact":p["v8_summary"]["payment_contact_suppression_pass"],
 "live cases explain decisions":all(x["reason_codes"] for x in p["v8_live_cases"]),
}
for k,v in checks.items():print(f'{"PASS" if v else "FAIL"}: {k}')
if not all(checks.values()):raise SystemExit(1)
