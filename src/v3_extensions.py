from __future__ import annotations
import json
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from .modeling import FEATURES, assign_decision, score_from_pd
from .scorecard import WOELogisticScorecard, metrics

ROOT=Path(__file__).resolve().parents[1]

def fairness_table(frame):
    parts=[]; frame=frame.copy(); frame["age_band"]=pd.cut(frame.age,[19,25,35,45,55,66],labels=["20-25","26-35","36-45","46-55","56-65"])
    for dimension in ["age_band","employment_type","channel"]:
        t=frame.groupby(dimension,observed=True).agg(applications=("default_12m","size"),approval_rate=("decision",lambda x:(x=="APPROVE").mean()),bad_rate=("default_12m","mean"),mean_pd=("pd_12m","mean")).reset_index().rename(columns={dimension:"group"}); ref=max(t.approval_rate.max(),1e-6); t["disparate_impact_ratio"]=t.approval_rate/ref; t["dimension"]=dimension; parts.append(t)
    return pd.concat(parts,ignore_index=True)

def behavioral_portfolio(n=2500,seed=77):
    r=np.random.default_rng(seed); util=np.clip(r.beta(2.2,2.6,n),0,1); pay=np.clip(r.normal(.88-.25*util,.18,n),0,1.2); dpd=r.choice([0,7,15,30,60,90],n,p=[.63,.11,.09,.08,.06,.03]); missed=np.clip(r.poisson(.35+1.7*(dpd>=30),n),0,8); cash=np.clip(r.normal(.12+.35*(dpd>=30),.18,n),-.4,1); bureau=np.clip(r.normal(640-55*(dpd>=30),65,n),300,850); z=-4+2.1*util-2.2*pay+.035*dpd+.33*missed+1.15*cash+.004*(620-bureau); pd90=1/(1+np.exp(-z)); default=r.binomial(1,pd90); risk=np.where((pd90>=.35)|(dpd>=60),"RED",np.where((pd90>=.15)|(dpd>=30),"AMBER","GREEN")); action=np.where(risk=="RED","Collections escalation",np.where(risk=="AMBER","Reduce limit and contact customer","Continue monitoring"))
    return pd.DataFrame({"account_id":[f"ACC-{i:06d}" for i in range(n)],"utilisation":util,"payment_ratio":pay,"days_past_due":dpd,"missed_payments_6m":missed,"cashflow_decline":cash,"bureau_score":bureau.astype(int),"behavioural_pd":pd90,"default_90d":default,"ews_status":risk,"recommended_action":action})

def build_v3():
    data=pd.read_csv(ROOT/"data/development_sample.csv"); train,test=data.iloc[:8000].copy(),data.iloc[8000:].copy()
    scorecard=WOELogisticScorecard().fit(train[FEATURES],train.default_12m); p=scorecard.predict_proba(test[FEATURES])[:,1]; test["pd_12m"]=p; test["credit_score"]=score_from_pd(p); test["decision"]=test.pd_12m.map(assign_decision); test["reason_codes"]=["; ".join(x) for x in scorecard.reason_codes(test[FEATURES])]
    sc_metrics=metrics(test.default_12m,p); sc_metrics.update({"model_id":"RCS_PD_3.0","validation_status":"PASS" if sc_metrics["auc"]>=.65 and sc_metrics["ks"]>=.25 and abs(sc_metrics["mean_pd"]-sc_metrics["bad_rate"])<.03 else "CONDITIONAL PASS"})
    # Accepted-only versus fuzzy-augmentation reject inference experiment.
    base=scorecard.predict_proba(train[FEATURES])[:,1]; accepted=(base<.22)&(train.existing_dti<.7); acc=train[accepted].copy(); rej=train[~accepted].copy(); acc_model=WOELogisticScorecard().fit(acc[FEATURES],acc.default_12m); p_acc=acc_model.predict_proba(test[FEATURES])[:,1]
    pr=acc_model.predict_proba(rej[FEATURES])[:,1]; aug=pd.concat([acc,rej.assign(default_12m=1),rej.assign(default_12m=0)],ignore_index=True); weights=np.r_[np.ones(len(acc)),pr,1-pr]; fuzzy=WOELogisticScorecard(); z=fuzzy.woe.fit_transform(aug[FEATURES],aug.default_12m); fuzzy.model.fit(z,aug.default_12m,sample_weight=weights); p_fuzzy=fuzzy.predict_proba(test[FEATURES])[:,1]
    reject=[{"method":"Full-information benchmark",**metrics(test.default_12m,p)},{"method":"Accepted-only",**metrics(test.default_12m,p_acc)},{"method":"Fuzzy augmentation",**metrics(test.default_12m,p_fuzzy)}]
    fair=fairness_table(test); behavior=behavioral_portfolio(); ews=behavior.groupby("ews_status").agg(accounts=("account_id","size"),observed_default_rate=("default_90d","mean"),mean_behavioural_pd=("behavioural_pd","mean"),mean_dpd=("days_past_due","mean"),mean_utilisation=("utilisation","mean")).reset_index()
    governance=[
      {"stage":"1. Business request","owner":"Retail Risk","status":"COMPLETE","evidence":"Use case, target and risk appetite"},{"stage":"2. Data approval","owner":"Data Governance","status":"COMPLETE","evidence":"Data dictionary and synthetic lineage"},{"stage":"3. Development","owner":"Model Development","status":"COMPLETE","evidence":"WOE scorecard and reject inference"},{"stage":"4. Independent validation","owner":"Model Risk","status":"COMPLETE","evidence":"Discrimination, calibration, stability, fairness"},{"stage":"5. UAT","owner":"Credit Operations","status":"READY","evidence":"API, reason codes and policy tests"},{"stage":"6. Committee approval","owner":"Model Risk Committee","status":"PENDING","evidence":"Approval memo and conditions"},{"stage":"7. Production release","owner":"Technology","status":"PENDING","evidence":"Release and rollback record"},{"stage":"8. Monitoring","owner":"Model Owner","status":"READY","evidence":"Monthly thresholds and actions"}]
    woe=[]
    for f in FEATURES:
        for b,v in scorecard.woe.maps_[f].items(): woe.append({"feature":f,"bin":b,"woe":v,"feature_iv":scorecard.woe.iv_[f],"coefficient":float(scorecard.model.coef_[0,FEATURES.index(f)]),"points":float(-50/np.log(2)*scorecard.model.coef_[0,FEATURES.index(f)]*v)})
    payload=json.loads((ROOT/"reports/validation_payload.json").read_text()); payload.update({"v3_scorecard_metrics":sc_metrics,"scorecard_table":woe,"reject_inference":reject,"fairness":fair.to_dict("records"),"ews_summary":ews.to_dict("records"),"governance":governance})
    (ROOT/"reports/validation_payload.json").write_text(json.dumps(payload,indent=2),encoding="utf-8"); behavior.to_csv(ROOT/"data/behavioural_ews_sample.csv",index=False); joblib.dump(scorecard,ROOT/"artifacts/retail_woe_scorecard.joblib"); (ROOT/"artifacts/model_metadata_v3.json").write_text(json.dumps(sc_metrics,indent=2),encoding="utf-8"); return sc_metrics

if __name__=="__main__": print(json.dumps(build_v3(),indent=2))
