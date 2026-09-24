from __future__ import annotations
import json
from pathlib import Path
import joblib, numpy as np, pandas as pd
from .modeling import FEATURES, assign_decision, score_from_pd
from .scorecard import WOELogisticScorecard, metrics

ROOT=Path(__file__).resolve().parents[1]

def pricing(pd12,income,requested,lgd=.65,cost_of_funds=.07,opex=350000):
    if pd12<.05: band,rate,mult="A",.16,6
    elif pd12<.10: band,rate,mult="B",.18,5
    elif pd12<.18: band,rate,mult="C",.22,3.5
    else: return {"risk_band":"D","decision":"DECLINE","interest_rate":None,"credit_limit":0,"expected_loss":0,"expected_profit":0,"raroc":0}
    limit=float(min(requested,income*mult)); el=pd12*lgd*limit; profit=rate*limit-el-cost_of_funds*limit-opex; capital=max(limit*(pd12*lgd*2+.04),1)
    return {"risk_band":band,"decision":"APPROVE" if pd12<.10 else "REVIEW","interest_rate":rate,"credit_limit":round(limit),"expected_loss":round(el),"expected_profit":round(profit),"raroc":profit/capital}

def build_v4():
    d=pd.read_csv(ROOT/"data/development_sample.csv",parse_dates=["application_date"]).sort_values("application_date").reset_index(drop=True); n=len(d); a,b=int(.6*n),int(.8*n); dev,val,oot=d.iloc[:a].copy(),d.iloc[a:b].copy(),d.iloc[b:].copy(); model=WOELogisticScorecard().fit(dev[FEATURES],dev.default_12m)
    windows=[]
    for name,f in [("Development",dev),("Validation",val),("Out-of-time",oot)]:
        p=model.predict_proba(f[FEATURES])[:,1]; q=metrics(f.default_12m,p); windows.append({"window":name,"start_date":str(f.application_date.min().date()),"end_date":str(f.application_date.max().date()),"applications":len(f),**q})
        if name=="Out-of-time": oot["pd_12m"]=p
    m=windows[-1].copy(); m.update({"model_id":"RCS_PD_4.0","validation_status":"PASS" if m["auc"]>=.65 and m["ks"]>=.25 and abs(m["mean_pd"]-m["bad_rate"])<.03 else "CONDITIONAL PASS"})
    oot["score"]=score_from_pd(oot.pd_12m); oot["decision"]=oot.pd_12m.map(assign_decision)
    strategy=[]
    for cutoff in np.arange(.04,.251,.01):
        f=oot[oot.pd_12m<cutoff]; napp=len(f); income=(f.requested_amount*.18).sum(); el=(f.pd_12m*.65*f.requested_amount).sum(); funding=(f.requested_amount*.07).sum(); op=napp*350000
        strategy.append({"cutoff":cutoff,"approved":napp,"approval_rate":napp/len(oot),"observed_bad_rate":f.default_12m.mean() if napp else 0,"interest_income":income,"expected_loss":el,"funding_cost":funding,"operating_cost":op,"expected_profit":income-el-funding-op})
    priced=[]
    for _,x in oot.iterrows(): priced.append({"application_id":x.application_id,"pd_12m":x.pd_12m,**pricing(x.pd_12m,x.monthly_income,x.requested_amount)})
    pricing_summary=pd.DataFrame(priced).groupby("risk_band").agg(applications=("application_id","size"),mean_pd=("pd_12m","mean"),approval_rate=("decision",lambda x:(x=="APPROVE").mean()),mean_rate=("interest_rate","mean"),total_limit=("credit_limit","sum"),expected_loss=("expected_loss","sum"),expected_profit=("expected_profit","sum"),mean_raroc=("raroc","mean")).reset_index()
    rng=np.random.default_rng(99); oot["month"]=oot.application_date.dt.to_period("M").astype(str); oot["fpd30"]=oot.default_12m*rng.binomial(1,.45,len(oot)); oot["mob3_30plus"]=np.maximum(oot.fpd30,oot.default_12m*rng.binomial(1,.68,len(oot))); vintage=oot.groupby("month").agg(bookings=("application_id","size"),approval_rate=("decision",lambda x:(x=="APPROVE").mean()),fpd30_rate=("fpd30","mean"),mob3_30plus_rate=("mob3_30plus","mean"),bad_rate_12m=("default_12m","mean"),mean_pd=("pd_12m","mean")).reset_index()
    review=oot[oot.decision!="APPROVE"].head(300).copy(); reasons=np.array(["Verified additional income","Bureau dispute resolved","Policy exception","Fraud concern","Affordability concern"]); review["override_flag"]=rng.random(len(review))<.18; review["override_direction"]=np.where(review.override_flag,np.where(review.decision=="DECLINE","DECLINE_TO_APPROVE","REVIEW_TO_DECLINE"),"NO_OVERRIDE"); review["override_reason"]=np.where(review.override_flag,rng.choice(reasons,len(review)),""); review["final_decision"]=np.where(review.override_direction=="DECLINE_TO_APPROVE","APPROVE",np.where(review.override_direction=="REVIEW_TO_DECLINE","DECLINE",review.decision)); review["authority"]="Senior Credit Officer"
    registry=[{"version":"RCS_PD_2.0","stage":"RETIRED","validation":"PASS","created":"2026-09-23","rollback_eligible":"YES"},{"version":"RCS_PD_3.0","stage":"STAGING","validation":"PASS","created":"2026-09-23","rollback_eligible":"YES"},{"version":"RCS_PD_4.0","stage":"PRODUCTION_CANDIDATE","validation":m["validation_status"],"created":"2026-09-23","rollback_eligible":"NO"}]
    scorecard=[]
    for f in FEATURES:
        coef=float(model.model.coef_[0,FEATURES.index(f)])
        for bin_name,woe in model.woe.maps_[f].items():scorecard.append({"feature":f,"bin":bin_name,"woe":woe,"feature_iv":model.woe.iv_[f],"coefficient":coef,"points":float(-50/np.log(2)*coef*woe),"monotonic_bins":len(model.woe.maps_[f])})
    pricing_records=pricing_summary.astype(object).where(pd.notnull(pricing_summary),None).to_dict("records")
    payload=json.loads((ROOT/"reports/validation_payload.json").read_text()); payload.update({"v4_metrics":m,"time_validation":windows,"v4_scorecard":scorecard,"v4_cutoff_strategy":strategy,"pricing_summary":pricing_records,"vintage":vintage.to_dict("records"),"override_summary":review.groupby(["override_direction","final_decision"]).agg(cases=("application_id","size"),mean_pd=("pd_12m","mean")).reset_index().to_dict("records"),"registry":registry})
    (ROOT/"reports/validation_payload.json").write_text(json.dumps(payload,indent=2),encoding="utf-8"); joblib.dump(model,ROOT/"artifacts/retail_woe_scorecard_v4.joblib"); (ROOT/"artifacts/model_metadata_v4.json").write_text(json.dumps(m,indent=2),encoding="utf-8"); pd.DataFrame(priced).to_csv(ROOT/"data/pricing_decisions_v4.csv",index=False); review.to_csv(ROOT/"data/manual_review_queue.csv",index=False); return m

if __name__=="__main__":print(json.dumps(build_v4(),indent=2))
