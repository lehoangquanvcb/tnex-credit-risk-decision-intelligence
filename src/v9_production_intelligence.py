from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
BUILD = "V9.0"
PORTFOLIO_CAPACITY = 1_000_000_000_000
NUMERIC = ["age","monthly_income","employment_months","bureau_score","existing_dti","requested_amount","tenor_months","inquiries_6m","identity_match_score","device_age_days","application_velocity_24h","failed_txn_rate","avg_balance_90d","transaction_count_90d","income_regularly","prior_loans"]
CATEGORICAL = ["employment_type","home_ownership","channel"]


def _ks(y, p):
    f=pd.DataFrame({"y":y,"p":p}).sort_values("p",ascending=False);bad=max(f.y.sum(),1);good=max((1-f.y).sum(),1)
    return float(np.max(np.abs(f.y.cumsum()/bad-(1-f.y).cumsum()/good)))


def train_product_models(data: pd.DataFrame):
    rows=[]; models={}
    for product,f in data.sort_values("application_date").groupby("product"):
        cut=max(int(len(f)*.75),1);train,test=f.iloc[:cut],f.iloc[cut:]
        prep=ColumnTransformer([("num",Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())]),NUMERIC),("cat",OneHotEncoder(handle_unknown="ignore"),CATEGORICAL)])
        pipe=Pipeline([("features",prep),("model",LogisticRegression(max_iter=700,C=.35,random_state=909))])
        pipe.fit(train[NUMERIC+CATEGORICAL],train.default_12m);pred=pipe.predict_proba(test[NUMERIC+CATEGORICAL])[:,1]
        auc=float(roc_auc_score(test.default_12m,pred));brier=float(brier_score_loss(test.default_12m,pred));gap=float(pred.mean()-test.default_12m.mean())
        status="PASS" if auc>=.62 and brier<=.20 and abs(gap)<=.08 else "REVIEW"
        rows.append({"product":product,"model_id":f"{product}_PD_2.0","train_rows":int(len(train)),"oot_rows":int(len(test)),"oot_auc":auc,"oot_gini":2*auc-1,"oot_ks":_ks(test.default_12m,pred),"oot_brier":brier,"mean_predicted_pd":float(pred.mean()),"observed_bad_rate":float(test.default_12m.mean()),"calibration_gap":gap,"status":status,"next_review":"2027-03-31"})
        models[product]=pipe;joblib.dump(pipe,ROOT/f"artifacts/{product.lower()}_pd_v9.joblib")
    return rows,models


def risk_appetite_optimizer(models: list[dict]):
    economics={"CASH_LOAN":{"yield":.275,"funding":.085,"lgd":.58,"opex":.018},"BNPL":{"yield":.205,"funding":.080,"lgd":.50,"opex":.025},"BUSINESS_LOAN":{"yield":.235,"funding":.090,"lgd":.48,"opex":.016}}
    observed={x["product"]:x["observed_bad_rate"] for x in models};rows=[]
    for cash in range(10,70,10):
        for bnpl in range(10,70-cash,10):
            business=100-cash-bnpl
            if business<10 or max(cash,bnpl,business)>60:continue
            alloc={"CASH_LOAN":cash/100,"BNPL":bnpl/100,"BUSINESS_LOAN":business/100}
            ecl=sum(PORTFOLIO_CAPACITY*w*observed[k]*economics[k]["lgd"] for k,w in alloc.items())
            revenue=sum(PORTFOLIO_CAPACITY*w*economics[k]["yield"] for k,w in alloc.items())
            funding=sum(PORTFOLIO_CAPACITY*w*economics[k]["funding"] for k,w in alloc.items())
            opex=sum(PORTFOLIO_CAPACITY*w*economics[k]["opex"] for k,w in alloc.items())
            contribution=revenue-funding-opex-ecl;capital=PORTFOLIO_CAPACITY*.12
            weighted_bad=sum(alloc[k]*observed[k] for k in alloc)
            passes=weighted_bad<=.13 and ecl<=75_000_000_000 and contribution>0
            rows.append({"cash_loan_share":alloc["CASH_LOAN"],"bnpl_share":alloc["BNPL"],"business_loan_share":alloc["BUSINESS_LOAN"],"portfolio_vnd":PORTFOLIO_CAPACITY,"weighted_bad_rate":weighted_bad,"expected_credit_loss_vnd":ecl,"net_contribution_vnd":contribution,"raroc":contribution/capital,"max_product_share":max(alloc.values()),"risk_appetite_status":"PASS" if passes else "BREACH"})
    eligible=[x for x in rows if x["risk_appetite_status"]=="PASS"]
    rec=max(eligible,key=lambda x:x["net_contribution_vnd"]) if eligible else max(rows,key=lambda x:x["net_contribution_vnd"])
    for x in rows:x["recommended"]=all(x[k]==rec[k] for k in ["cash_loan_share","bnpl_share","business_loan_share"])
    return rows,rec


def governance_and_outcomes(data, model_rows):
    governance=[]
    for i,m in enumerate(model_rows,1):
        governance.append({"model_id":m["model_id"],"model":f'{m["product"].replace("_"," ").title()} application PD',"materiality":"HIGH","owner":"Head of Credit Risk Modeling","independent_validator":"Model Validation","status":m["status"],"last_validation":"2026-09-24","next_review":m["next_review"],"open_findings":1 if m["status"]=="REVIEW" else 0,"deployment":"SHADOW","approval_body":"Model Risk Committee"})
    governance += [
        {"model_id":"FRAUD_GRAPH_1.0","model":"Linked-identity fraud network","materiality":"HIGH","owner":"Fraud Risk","independent_validator":"Financial Crime Validation","status":"PASS","last_validation":"2026-09-24","next_review":"2027-03-31","open_findings":1,"deployment":"CANARY","approval_body":"Financial Crime Committee"},
        {"model_id":"COLLECTIONS_NBA_1.0","model":"Collections next-best-action","materiality":"MEDIUM","owner":"Collections","independent_validator":"Customer Outcomes","status":"PASS","last_validation":"2026-09-24","next_review":"2027-09-30","open_findings":0,"deployment":"SHADOW","approval_body":"Customer Committee"}]
    findings=[
        {"finding_id":"V9-F01","model_id":"FRAUD_GRAPH_1.0","severity":"MEDIUM","finding":"Validate linked-device false-positive rate on confirmed outcomes","owner":"Fraud Analytics","due_date":"2026-12-15","status":"OPEN"},
        {"finding_id":"V9-F02","model_id":"BUSINESS_LOAN_PD_2.0","severity":"MEDIUM","finding":"Expand sales-data history before production calibration","owner":"Business Credit","due_date":"2027-01-31","status":"OPEN"},
        {"finding_id":"V9-F03","model_id":"ALL_PRODUCT_PD","severity":"LOW","finding":"Complete quarterly reason-code stability review","owner":"Model Monitoring","due_date":"2026-12-31","status":"IN PROGRESS"}]
    x=data.copy();x["income_segment"]=pd.qcut(x.monthly_income,3,labels=["Lower","Middle","Higher"]);x["repeat_borrower"]=x.prior_loans>0;x["high_debt_burden"]=x.existing_dti>=.60;x["affordability_pass"]=x.existing_dti<=.65;x["complaint_proxy"]=(x.failed_txn_rate>.15)|(x.identity_match_score<.65);x["hardship_proxy"]=(x.existing_dti>.75)&(x.default_12m==1)
    outcomes=[]
    for (product,segment),f in x.groupby(["product","income_segment"],observed=True):
        outcomes.append({"product":product,"income_segment":str(segment),"customers":int(len(f)),"approval_rate":float((f.decision=="APPROVE").mean()),"observed_bad_rate":float(f.default_12m.mean()),"mean_dti":float(f.existing_dti.mean()),"high_debt_burden_rate":float(f.high_debt_burden.mean()),"repeat_borrower_rate":float(f.repeat_borrower.mean()),"complaint_proxy_rate":float(f.complaint_proxy.mean()),"hardship_proxy_rate":float(f.hardship_proxy.mean()),"affordability_pass_rate":float(f.affordability_pass.mean())})
    controls=[
        {"control":"Affordability pass rate","value":float(x.affordability_pass.mean()),"threshold":.85,"direction":"MIN","status":"PASS" if x.affordability_pass.mean()>=.85 else "REVIEW","owner":"Credit Policy"},
        {"control":"High debt-burden approval rate","value":float((x.loc[x.high_debt_burden,"decision"]=="APPROVE").mean()),"threshold":.45,"direction":"MAX","status":"PASS" if (x.loc[x.high_debt_burden,"decision"]=="APPROVE").mean()<=.45 else "REVIEW","owner":"Responsible Lending"},
        {"control":"Complaint proxy rate","value":float(x.complaint_proxy.mean()),"threshold":.05,"direction":"MAX","status":"PASS" if x.complaint_proxy.mean()<=.05 else "REVIEW","owner":"Customer Experience"},
        {"control":"Hardship proxy rate","value":float(x.hardship_proxy.mean()),"threshold":.03,"direction":"MAX","status":"PASS" if x.hardship_proxy.mean()<=.03 else "REVIEW","owner":"Collections"},
        {"control":"Payment-pending contact suppression","value":1.0,"threshold":1.0,"direction":"MIN","status":"PASS","owner":"Operations"}]
    return governance,findings,outcomes,controls


def build():
    data=pd.read_csv(ROOT/"data/tnex_decisions_v7.csv",parse_dates=["application_date"])
    model_rows,_=train_product_models(data);allocations,recommended=risk_appetite_optimizer(model_rows);governance,findings,outcomes,controls=governance_and_outcomes(data,model_rows)
    committee=[
        {"section":"Portfolio recommendation","decision":f'Allocate {recommended["cash_loan_share"]:.0%} Cash Loan, {recommended["bnpl_share"]:.0%} BNPL and {recommended["business_loan_share"]:.0%} Business Loan',"metric":recommended["net_contribution_vnd"],"status":"RECOMMEND","owner":"CRO"},
        {"section":"Model approval","decision":"Retain all product models in shadow mode pending live-data validation","metric":sum(x["status"]=="PASS" for x in model_rows),"status":"CONDITIONAL","owner":"Model Risk Committee"},
        {"section":"Fraud control","decision":"Continue canary with linked-device false-positive review","metric":1,"status":"ACTION","owner":"Fraud Risk"},
        {"section":"Customer outcomes","decision":"Review any responsible-lending metric outside threshold monthly","metric":sum(x["status"]!="PASS" for x in controls),"status":"MONITOR","owner":"Customer Committee"}]
    summary={"build":BUILD,"platform":"TNEX-aligned production credit risk and decision intelligence simulation","calculated_at":datetime.now(timezone.utc).isoformat(),"product_models":3,"models_passing":sum(x["status"]=="PASS" for x in model_rows),"portfolio_capacity_vnd":PORTFOLIO_CAPACITY,"recommended_cash_share":recommended["cash_loan_share"],"recommended_bnpl_share":recommended["bnpl_share"],"recommended_business_share":recommended["business_loan_share"],"recommended_ecl_vnd":recommended["expected_credit_loss_vnd"],"recommended_contribution_vnd":recommended["net_contribution_vnd"],"recommended_raroc":recommended["raroc"],"open_findings":sum(x["status"]!="CLOSED" for x in findings),"customer_controls_passed":sum(x["status"]=="PASS" for x in controls),"overall_status":"CONDITIONAL PASS — SIMULATION","disclaimer":"Synthetic portfolio only; not an official TNEX model or affiliated implementation."}
    path=ROOT/"reports/validation_payload.json";payload=json.loads(path.read_text());payload.update({"v9_summary":summary,"v9_product_models":model_rows,"v9_risk_appetite_grid":allocations,"v9_recommended_allocation":recommended,"v9_model_inventory":governance,"v9_validation_findings":findings,"v9_customer_outcomes":outcomes,"v9_customer_controls":controls,"v9_committee_pack":committee});path.write_text(json.dumps(payload,indent=2,allow_nan=False),encoding="utf-8")
    (ROOT/"artifacts/model_metadata_v9.json").write_text(json.dumps(summary,indent=2),encoding="utf-8");pd.DataFrame(allocations).to_csv(ROOT/"data/portfolio_allocation_v9.csv",index=False);pd.DataFrame(governance).to_csv(ROOT/"data/model_inventory_v9.csv",index=False);pd.DataFrame(outcomes).to_csv(ROOT/"data/customer_outcomes_v9.csv",index=False)
    return summary


if __name__=="__main__":print(json.dumps(build(),indent=2))
