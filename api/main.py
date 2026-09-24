from pathlib import Path
import json, sys, uuid, joblib, pandas as pd
from fastapi import FastAPI
from fastapi import Depends,HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.modeling import FEATURES, assign_decision, score_from_pd
from src.policy import decide
from src.audit import apply_override,find_audit,log_decision
from src.auth import require_roles
from src.v8_digital_lending_os import live_decision

app = FastAPI(title="Retail Credit PD API", version="9.0")
model = joblib.load(ROOT / "artifacts/retail_woe_scorecard_v4.joblib")
validation_payload = json.loads((ROOT / "reports/validation_payload.json").read_text())

class Application(BaseModel):
    application_id: str | None = None
    age: int = Field(ge=20, le=65); monthly_income: float = Field(gt=0); employment_months: int = Field(ge=0)
    bureau_score: int = Field(ge=300, le=850); existing_dti: float = Field(ge=0, le=1); requested_amount: float = Field(gt=0)
    tenor_months: int; inquiries_6m: int = Field(ge=0); employment_type: str; home_ownership: str; channel: str
class OverrideRequest(BaseModel):
    audit_id: str; final_decision: str; reason: str
class DigitalApplication(Application):
    product: str = "CASH_LOAN"
    identity_match_score: float = Field(default=.90, ge=0, le=1); device_age_days: int = Field(default=180, ge=0)
    application_velocity_24h: int = Field(default=0, ge=0); failed_txn_rate: float = Field(default=.02, ge=0, le=1)
    transaction_count_90d: int = Field(default=35, ge=0); income_regularly: int = Field(default=1, ge=0, le=1)
    prior_on_time_ratio: float = Field(default=.85, ge=0, le=1); sales_monthly: float = Field(default=0, ge=0)
    shared_device_accounts: int = Field(default=0, ge=0); shared_payout_accounts: int = Field(default=0, ge=0)

@app.get("/health")
def health(): return {"status":"ok", "model_id":"RCS_PD_4.0", "model_type":"Monotonic WOE logistic scorecard"}

@app.post("/score")
def score(application: Application,actor=Depends(require_roles("credit_officer","model_owner","admin"))):
    raw=application.model_dump(); application_id=raw.pop("application_id") or f"API-{uuid.uuid4().hex[:12]}"; frame = pd.DataFrame([raw])[FEATURES]
    pd12 = float(model.predict_proba(frame)[0,1])
    offer=decide(pd12,application.monthly_income,application.requested_amount,application.existing_dti,application.tenor_months); output={"application_id":application_id,"pd_12m":round(pd12,6),"credit_score":int(score_from_pd([pd12])[0]),"decision":offer["decision"],"reason_codes":model.reason_codes(frame)[0],**offer,"model_id":"RCS_PD_4.0"}; output["audit_id"]=log_decision(application_id,"RCS_PD_4.0",raw,output,actor["sub"]); return output

@app.post("/override")
def override(request:OverrideRequest,actor=Depends(require_roles("approver","admin"))):
    if request.final_decision not in ["APPROVE","DECLINE","REVIEW"]:raise HTTPException(400,"Invalid final decision")
    count=apply_override(request.audit_id,request.model_dump(exclude={"audit_id"}),actor["sub"])
    if not count:raise HTTPException(404,"Audit record not found")
    return {"status":"recorded","audit_id":request.audit_id,"approved_by":actor["sub"]}

@app.get("/audit")
def audit(application_id:str|None=None,limit:int=100,actor=Depends(require_roles("auditor","admin"))):return find_audit(application_id,limit)

@app.get("/portfolio/v5/summary")
def portfolio_summary(actor=Depends(require_roles("model_owner","auditor","admin"))):
    return validation_payload["v5_summary"]

@app.get("/portfolio/v5/stress")
def portfolio_stress(actor=Depends(require_roles("model_owner","auditor","admin"))):
    return validation_payload["v5_stress_testing"]

@app.get("/portfolio/v6/ifrs9")
def ifrs9_summary(actor=Depends(require_roles("model_owner","auditor","admin"))):
    return {"summary":validation_payload["v6_summary"],"stages":validation_payload["v6_ifrs9_stage_summary"],"scenarios":validation_payload["v6_macro_scenarios"]}

@app.get("/portfolio/v6/monitoring")
def delayed_monitoring(actor=Depends(require_roles("model_owner","auditor","admin"))):
    return validation_payload["v6_delayed_monitoring"]

@app.get("/portfolio/v6/deployment")
def deployment_status(actor=Depends(require_roles("model_owner","auditor","admin"))):
    return {"gates":validation_payload["v6_deployment_gates"],"incidents":validation_payload["v6_incidents"],"rollback":validation_payload["v6_rollback_drill"]}

@app.get("/tnex/v7/products")
def tnex_products(actor=Depends(require_roles("model_owner","auditor","admin"))):
    return {"summary": validation_payload["v7_summary"], "configuration": validation_payload["v7_product_config"], "performance": validation_payload["v7_product_performance"]}

@app.get("/tnex/v7/decision-health")
def tnex_decision_health(actor=Depends(require_roles("model_owner","auditor","admin"))):
    return {"latency": validation_payload["v7_latency"], "features": validation_payload["v7_feature_governance"]}

@app.get("/tnex/v7/funnel")
def tnex_funnel(actor=Depends(require_roles("model_owner","auditor","admin"))):
    return validation_payload["v7_funnel"]

@app.get("/tnex/v7/lifecycle")
def tnex_lifecycle(actor=Depends(require_roles("model_owner","auditor","admin"))):
    return {"repeat_customer_actions": validation_payload["v7_lifecycle"], "collections_controls": validation_payload["v7_collections"]}

@app.post("/tnex/v8/live-decision")
def tnex_live_decision(application:DigitalApplication,actor=Depends(require_roles("credit_officer","model_owner","admin"))):
    raw=application.model_dump(); output=live_decision(raw,model); output["audit_id"]=log_decision(output["application_id"],output["model_id"],raw,output,actor["sub"]); return output

@app.get("/tnex/v8/optimizer")
def tnex_optimizer(actor=Depends(require_roles("model_owner","auditor","admin"))):
    return {"recommended":validation_payload["v8_recommended_strategy"],"strategies":validation_payload["v8_optimizer"]}

@app.get("/tnex/v8/fraud-network")
def tnex_fraud_network(actor=Depends(require_roles("model_owner","auditor","admin"))): return validation_payload["v8_fraud_network"]

@app.get("/tnex/v8/collections-nba")
def tnex_collections_nba(actor=Depends(require_roles("model_owner","auditor","admin"))): return validation_payload["v8_collections_nba"]

@app.get("/tnex/v8/command-center")
def tnex_command_center(actor=Depends(require_roles("model_owner","auditor","admin"))):
    return {"summary":validation_payload["v8_summary"],"monitoring":validation_payload["v8_monitoring"],"product_models":validation_payload["v8_product_models"]}

@app.get("/tnex/v9/product-models")
def v9_product_models(actor=Depends(require_roles("model_owner","auditor","admin"))): return validation_payload["v9_product_models"]

@app.get("/tnex/v9/risk-appetite")
def v9_risk_appetite(actor=Depends(require_roles("model_owner","auditor","admin"))):
    return {"summary":validation_payload["v9_summary"],"recommended":validation_payload["v9_recommended_allocation"],"grid":validation_payload["v9_risk_appetite_grid"]}

@app.get("/tnex/v9/governance")
def v9_governance(actor=Depends(require_roles("model_owner","auditor","admin"))):
    return {"inventory":validation_payload["v9_model_inventory"],"findings":validation_payload["v9_validation_findings"]}

@app.get("/tnex/v9/customer-outcomes")
def v9_customer_outcomes(actor=Depends(require_roles("model_owner","auditor","admin"))):
    return {"controls":validation_payload["v9_customer_controls"],"segments":validation_payload["v9_customer_outcomes"]}

@app.get("/tnex/v9/risk-committee-pack")
def v9_committee_pack(actor=Depends(require_roles("model_owner","auditor","admin"))): return validation_payload["v9_committee_pack"]
