from __future__ import annotations
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
import joblib,numpy as np,pandas as pd
from .modeling import FEATURES,score_from_pd
from .policy import decide,load_policy
from .scorecard import WOELogisticScorecard,metrics

ROOT=Path(__file__).resolve().parents[1];MODEL_ID="RCS_PD_4.0";DATASET_ID="SYNTH_RETAIL_2024_2025";BUILD="V4.1"
def stamp(rows):
    ts=datetime.now(timezone.utc).isoformat();return [dict(x,model_id=MODEL_ID,dataset_id=DATASET_ID,build=BUILD,calculated_at=ts) for x in rows]
def fairness(frame):
    f=frame.copy();f["age_band"]=pd.cut(f.age,[19,25,35,45,55,66],labels=["20-25","26-35","36-45","46-55","56-65"]);out=[]
    for dim in ["age_band","employment_type","channel"]:
        t=f.groupby(dim,observed=True).agg(applications=("default_12m","size"),approval_rate=("decision",lambda x:(x=="APPROVE").mean()),bad_rate=("default_12m","mean"),mean_pd=("pd_12m","mean")).reset_index().rename(columns={dim:"group"});ref=max(t.approval_rate.max(),1e-9);t["disparate_impact_ratio"]=t.approval_rate/ref;t["small_sample_warning"]=t.applications<100;t["dimension"]=dim;out+=t.to_dict("records")
    return out
def build():
    d=pd.read_csv(ROOT/"data/development_sample.csv",parse_dates=["application_date"]).sort_values("application_date").reset_index(drop=True);a,b=int(.6*len(d)),int(.8*len(d));dev,val,oot=d.iloc[:a].copy(),d.iloc[a:b].copy(),d.iloc[b:].copy();model=joblib.load(ROOT/"artifacts/retail_woe_scorecard_v4.joblib");policy=load_policy()
    for f in [dev,val,oot]:
        f["pd_12m"]=model.predict_proba(f[FEATURES])[:,1];offers=[decide(x.pd_12m,x.monthly_income,x.requested_amount,x.existing_dti,x.tenor_months,policy) for _,x in f.iterrows()];f["decision"]=[x["decision"] for x in offers];f["score"]=score_from_pd(f.pd_12m)
    oot["risk_band"]=pd.qcut(oot.pd_12m,10,duplicates="drop").astype(str);cal=oot.groupby("risk_band",observed=True).agg(applications=("default_12m","size"),observed_bad_rate=("default_12m","mean"),predicted_pd=("pd_12m","mean"),average_score=("score","mean")).reset_index().to_dict("records")
    oot["month"]=oot.application_date.dt.to_period("M").astype(str);mon=oot.groupby("month").agg(applications=("default_12m","size"),observed_bad_rate=("default_12m","mean"),mean_pd=("pd_12m","mean"),approval_rate=("decision",lambda x:(x=="APPROVE").mean()),mean_score=("score","mean")).reset_index().to_dict("records")
    base=model.predict_proba(dev[FEATURES])[:,1];accepted=(base<policy["review_pd_max"])&(dev.existing_dti<=policy["maximum_dti"]);acc,rej=dev[accepted].copy(),dev[~accepted].copy();accm=WOELogisticScorecard().fit(acc[FEATURES],acc.default_12m);pa=accm.predict_proba(oot[FEATURES])[:,1];pr=accm.predict_proba(rej[FEATURES])[:,1];aug=pd.concat([acc,rej.assign(default_12m=1),rej.assign(default_12m=0)],ignore_index=True);w=np.r_[np.ones(len(acc)),pr,1-pr];fuzzy=WOELogisticScorecard();z=fuzzy.woe.fit_transform(aug[FEATURES],aug.default_12m);fuzzy.model.fit(z,aug.default_12m,sample_weight=w);pf=fuzzy.predict_proba(oot[FEATURES])[:,1];reject=[{"method":"Full-information benchmark",**metrics(oot.default_12m,oot.pd_12m)},{"method":"Accepted-only",**metrics(oot.default_12m,pa)},{"method":"Fuzzy augmentation",**metrics(oot.default_12m,pf)}]
    stability=[]
    for feature in FEATURES:
        if feature in model.woe.edges_:kd=pd.cut(dev[feature],model.woe.edges_[feature],include_lowest=True).astype(str);ko=pd.cut(oot[feature],model.woe.edges_[feature],include_lowest=True).astype(str)
        else:kd=dev[feature].fillna("MISSING").astype(str);ko=oot[feature].fillna("MISSING").astype(str)
        keys=sorted(set(kd)|set(ko));fd=kd.value_counts(normalize=True);fo=ko.value_counts(normalize=True)
        for key in keys:
            dp=max(fd.get(key,0),1e-6);op=max(fo.get(key,0),1e-6);idxd=kd==key;idxo=ko==key;stability.append({"feature":feature,"bin":key,"development_share":dp,"oot_share":op,"psi_component":(op-dp)*np.log(op/dp),"development_bad_rate":dev.loc[idxd,"default_12m"].mean() if idxd.any() else None,"oot_bad_rate":oot.loc[idxo,"default_12m"].mean() if idxo.any() else None,"woe":model.woe.maps_[feature].get(key,0)})
    z=model.woe.transform(oot[FEATURES].head(250));logit=model.model.intercept_[0]+z.to_numpy()@model.model.coef_[0];manual_pd=1/(1+np.exp(-logit));api_pd=model.predict_proba(oot[FEATURES].head(250))[:,1];factor=50/np.log(2);raw_score=600+factor*(np.log((1-api_pd)/api_pd)-np.log(20));reconstructed_pd=1/(1+20*np.exp((raw_score-600)/factor));recon={"sample_size":250,"max_model_vs_logit_pd_gap":float(np.max(np.abs(api_pd-manual_pd))),"max_score_to_pd_gap":float(np.max(np.abs(api_pd-reconstructed_pd))),"tolerance":1e-6,"status":"PASS" if max(np.max(np.abs(api_pd-manual_pd)),np.max(np.abs(api_pd-reconstructed_pd)))<1e-6 else "FAIL"}
    payload=json.loads((ROOT/"reports/validation_payload.json").read_text());payload.update({"v41_calibration":stamp(cal),"v41_monitoring":stamp(mon),"v41_fairness":stamp(fairness(oot)),"v41_reject_inference":stamp(reject),"bin_stability":stamp(stability),"reconstruction":recon,"policy":policy,"v41_metadata":{"model_id":MODEL_ID,"policy_version":policy["policy_version"],"dataset_id":DATASET_ID,"build":BUILD,"code_hash":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12],"calculated_at":datetime.now(timezone.utc).isoformat()}});(ROOT/"reports/validation_payload.json").write_text(json.dumps(payload,indent=2,allow_nan=False));return payload["v41_metadata"]|recon
if __name__=="__main__":print(json.dumps(build(),indent=2))
