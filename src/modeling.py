from __future__ import annotations
import json
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT=Path(__file__).resolve().parents[1]
NUMERIC=["age","monthly_income","employment_months","bureau_score","existing_dti","requested_amount","tenor_months","inquiries_6m"]
CATEGORICAL=["employment_type","home_ownership","channel"]
FEATURES=NUMERIC+CATEGORICAL

def generate_portfolio(n=12000,seed=42,drift=0.0,start="2024-01-01"):
    r=np.random.default_rng(seed); age=np.clip(r.normal(35-drift,9,n),20,65).round(); income=np.exp(r.normal(np.log(18e6*(1-.08*drift)),.55,n)); employment=np.clip(r.gamma(3.2,18,n),0,360).round(); bureau=np.clip(r.normal(640-18*drift,72,n),300,850).round(); dti=np.clip(r.beta(2.1+.2*drift,4.5,n),.01,.95); amount=np.exp(r.normal(np.log(55e6*(1+.08*drift)),.65,n)); tenor=r.choice([6,9,12,18,24,36],n,p=[.08,.08,.29,.16,.27,.12]); inquiries=np.clip(r.poisson(1.2+.35*drift,n),0,10); emp=r.choice(["Salaried","Self-employed","Contract"],n,p=[.64,.23,.13]); home=r.choice(["Owned","Mortgage","Rented","Family"],n,p=[.25,.18,.37,.20]); channel=r.choice(["Mobile","Branch","Partner"],n,p=[.58,.17,.25])
    z=-2.7+1.65*dti+.0058*(620-bureau)+.19*inquiries+6e-9*amount-2.5e-8*income-.0025*employment+.35*(emp=="Contract")+.28*(emp=="Self-employed")+.25*(home=="Rented")+.18*(channel=="Partner")+.22*drift
    return pd.DataFrame({"application_id":[f"APP-{seed}-{i:06d}" for i in range(n)],"application_date":pd.date_range(start,periods=n,freq="h"),"age":age.astype(int),"monthly_income":income.round(),"employment_months":employment.astype(int),"bureau_score":bureau.astype(int),"existing_dti":dti.round(4),"requested_amount":amount.round(),"tenor_months":tenor,"inquiries_6m":inquiries,"employment_type":emp,"home_ownership":home,"channel":channel,"default_12m":r.binomial(1,1/(1+np.exp(-z)))})

def preprocessing(dense=False):
    num=Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())]); cat=Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore",sparse_output=not dense))])
    return ColumnTransformer([("num",num,NUMERIC),("cat",cat,CATEGORICAL)])
def build_logistic(): return Pipeline([("prep",preprocessing()),("model",LogisticRegression(max_iter=1500,class_weight="balanced",C=.35))])
def build_calibrated(): return CalibratedClassifierCV(build_logistic(),method="sigmoid",cv=5)
def build_challenger(): return Pipeline([("prep",preprocessing(True)),("model",RandomForestClassifier(n_estimators=220,min_samples_leaf=35,max_depth=9,class_weight="balanced",random_state=42,n_jobs=-1))])
def ks_stat(y,p):
    fpr,tpr,_=roc_curve(y,p); return float(np.max(tpr-fpr))
def psi(e,a,bins=10):
    cuts=np.unique(np.quantile(e,np.linspace(0,1,bins+1))); cuts[0],cuts[-1]=-np.inf,np.inf; ep=np.clip(np.histogram(e,bins=cuts)[0]/len(e),1e-6,None); ap=np.clip(np.histogram(a,bins=cuts)[0]/len(a),1e-6,None); return float(np.sum((ap-ep)*np.log(ap/ep)))
def score_from_pd(p):
    p=np.asarray(p); return np.clip(600+50/np.log(2)*np.log(((1-p)/np.clip(p,1e-8,1))/20),300,850).round().astype(int)
def assign_decision(p): return "APPROVE" if p<.08 else "REVIEW" if p<.18 else "DECLINE"
def reason_codes(x,limit=3):
    c=[]
    if x.existing_dti>=.55:c.append((x.existing_dti,"HIGH_EXISTING_DTI"))
    if x.bureau_score<580:c.append(((580-x.bureau_score)/100,"LOW_BUREAU_SCORE"))
    if x.inquiries_6m>=3:c.append((x.inquiries_6m/5,"MULTIPLE_RECENT_INQUIRIES"))
    if x.employment_months<12:c.append(((12-x.employment_months)/12,"SHORT_EMPLOYMENT_HISTORY"))
    if x.requested_amount/max(x.monthly_income,1)>5:c.append((x.requested_amount/max(x.monthly_income,1)/10,"HIGH_LOAN_TO_INCOME"))
    if x.employment_type=="Contract":c.append((.35,"CONTRACT_EMPLOYMENT"))
    if x.home_ownership=="Rented":c.append((.2,"RENTED_RESIDENCE"))
    return [v for _,v in sorted(c,reverse=True)[:limit]] or ["NO_MATERIAL_ADVERSE_FACTOR"]

def woe_iv_table(f):
    rows=[]; tg=max((f.default_12m==0).sum(),1); tb=max((f.default_12m==1).sum(),1)
    for feature in FEATURES:
        if feature in NUMERIC:
            intervals=pd.qcut(f[feature],5,duplicates="drop")
            decimals=0 if feature in ["age","monthly_income","employment_months","bureau_score","requested_amount","tenor_months","inquiries_6m"] else 2
            key=intervals.map(lambda v:f"{v.left:,.{decimals}f} to {v.right:,.{decimals}f}").astype(str)
        else: key=f[feature].astype(str)
        t=f.assign(_bin=key).groupby("_bin",observed=True).default_12m.agg(["count","sum"]).reset_index(); t["good"]=t["count"]-t["sum"]; t["dg"]=(t.good+.5)/(tg+.5*len(t)); t["db"]=(t["sum"]+.5)/(tb+.5*len(t)); t["woe"]=np.log(t.dg/t.db); t["iv_component"]=(t.dg-t.db)*t.woe
        for _,z in t.iterrows(): rows.append({"feature":feature,"bin":z._bin,"count":int(z["count"]),"bad_rate":z["sum"]/z["count"],"woe":z.woe,"iv_component":z.iv_component})
    out=pd.DataFrame(rows); out["feature_iv"]=out.groupby("feature").iv_component.transform("sum"); return out

def train_and_validate():
    data=generate_portfolio(); oot=generate_portfolio(3000,2026,.55,"2025-01-01"); train,test=data.iloc[:8000].copy(),data.iloc[8000:].copy(); raw,model,challenger=build_logistic(),build_calibrated(),build_challenger(); raw.fit(train[FEATURES],train.default_12m); model.fit(train[FEATURES],train.default_12m); challenger.fit(train[FEATURES],train.default_12m)
    pt=model.predict_proba(train[FEATURES])[:,1]; pv=model.predict_proba(test[FEATURES])[:,1]; po=model.predict_proba(oot[FEATURES])[:,1]; comparison=[]
    for name,m in [("Raw logistic",raw),("Calibrated logistic (champion)",model),("Random forest challenger",challenger)]:
        p=m.predict_proba(test[FEATURES])[:,1]; a=roc_auc_score(test.default_12m,p); comparison.append({"model":name,"auc":a,"gini":2*a-1,"ks":ks_stat(test.default_12m,p),"brier":brier_score_loss(test.default_12m,p)})
    metrics={"model_id":"RCS_PD_2.0","model_type":"Calibrated 12-month application PD","target":"default_12m","development_rows":len(train),"holdout_rows":len(test),"oot_rows":len(oot),"train_auc":roc_auc_score(train.default_12m,pt),"test_auc":roc_auc_score(test.default_12m,pv),"oot_auc":roc_auc_score(oot.default_12m,po),"test_gini":2*roc_auc_score(test.default_12m,pv)-1,"test_ks":ks_stat(test.default_12m,pv),"test_brier":brier_score_loss(test.default_12m,pv),"score_psi_oot":psi(pt,po),"test_bad_rate":test.default_12m.mean(),"test_mean_pd":pv.mean(),"oot_bad_rate":oot.default_12m.mean(),"oot_mean_pd":po.mean()}; metrics["validation_status"]="PASS" if metrics["test_auc"]>=.65 and metrics["test_ks"]>=.25 and metrics["score_psi_oot"]<.25 and abs(metrics["test_mean_pd"]-metrics["test_bad_rate"])<.03 else "CONDITIONAL PASS"
    for f,p in [(test,pv),(oot,po)]: f["pd_12m"]=p; f["credit_score"]=score_from_pd(p); f["decision"]=f.pd_12m.map(assign_decision); f["reason_codes"]=f.apply(lambda x:"; ".join(reason_codes(x)),axis=1)
    calibration=test.assign(risk_band=pd.qcut(test.pd_12m,10,duplicates="drop").astype(str)).groupby("risk_band",observed=True).agg(applications=("default_12m","size"),observed_bad_rate=("default_12m","mean"),predicted_pd=("pd_12m","mean"),avg_score=("credit_score","mean")).reset_index()
    cutoffs=[]
    for cutoff in np.arange(.04,.251,.01):
        a=test[test.pd_12m<cutoff]; n=len(a); revenue=(a.requested_amount*.18).sum(); el=(a.pd_12m*.65*a.requested_amount).sum(); funding=(a.requested_amount*.07).sum(); opex=n*350000; cutoffs.append({"cutoff":cutoff,"approved":n,"approval_rate":n/len(test),"observed_bad_rate":a.default_12m.mean() if n else 0,"interest_income":revenue,"expected_loss":el,"funding_cost":funding,"operating_cost":opex,"expected_profit":revenue-el-funding-opex})
    oot["month"]=pd.to_datetime(oot.application_date).dt.to_period("M").astype(str); monitoring=oot.groupby("month").agg(applications=("default_12m","size"),observed_bad_rate=("default_12m","mean"),mean_pd=("pd_12m","mean"),approval_rate=("decision",lambda x:(x=="APPROVE").mean()),mean_score=("credit_score","mean")).reset_index()
    payload={"metrics":metrics,"model_comparison":comparison,"calibration":calibration.to_dict("records"),"cutoff_strategy":cutoffs,"woe_iv":woe_iv_table(train).to_dict("records"),"monitoring":monitoring.to_dict("records")}
    for d in [ROOT/"data",ROOT/"artifacts",ROOT/"reports"]:d.mkdir(exist_ok=True)
    data.to_csv(ROOT/"data/development_sample.csv",index=False); oot.to_csv(ROOT/"data/production_monitoring_sample.csv",index=False); joblib.dump(model,ROOT/"artifacts/retail_pd_model.joblib"); (ROOT/"artifacts/model_metadata.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8"); (ROOT/"reports/validation_payload.json").write_text(json.dumps(payload,indent=2),encoding="utf-8"); return metrics

if __name__=="__main__":print(json.dumps(train_and_validate(),indent=2))
