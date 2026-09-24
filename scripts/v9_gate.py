import json
from pathlib import Path
p=json.loads((Path(__file__).resolve().parents[1]/"reports/validation_payload.json").read_text())
r=p["v9_recommended_allocation"]
checks={"three independent models":len(p["v9_product_models"])==3,"allocation equals 100%":abs(r["cash_loan_share"]+r["bnpl_share"]+r["business_loan_share"]-1)<1e-9,"risk appetite passes":r["risk_appetite_status"]=="PASS","concentration limit passes":r["max_product_share"]<=.60,"governance ownership complete":all(x["owner"] and x["independent_validator"] for x in p["v9_model_inventory"]),"customer controls present":len(p["v9_customer_controls"])>=5}
for k,v in checks.items():print(f'{"PASS" if v else "FAIL"}: {k}')
if not all(checks.values()):raise SystemExit(1)
