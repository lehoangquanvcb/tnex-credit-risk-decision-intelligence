from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve

NUMERIC=["age","monthly_income","employment_months","bureau_score","existing_dti","requested_amount","tenor_months","inquiries_6m"]
CATEGORICAL=["employment_type","home_ownership","channel"]
FEATURES=NUMERIC+CATEGORICAL

class WOETransformer(BaseEstimator,TransformerMixin):
    def __init__(self,n_bins=8,smoothing=.5): self.n_bins=n_bins; self.smoothing=smoothing
    def _monotonic_edges(self,x,y):
        x=pd.to_numeric(x).fillna(pd.to_numeric(x).median()); edges=np.unique(np.quantile(x,np.linspace(0,1,self.n_bins+1)))
        if len(edges)<3:return np.array([-np.inf,np.inf])
        edges[0],edges[-1]=-np.inf,np.inf
        while len(edges)>3:
            key=pd.cut(x,edges,include_lowest=True); rates=pd.DataFrame({"key":key,"y":y}).groupby("key",observed=True).y.mean().to_numpy()
            if len(rates)<2:break
            direction=1 if np.corrcoef(np.arange(len(rates)),rates)[0,1]>=0 else -1; bad=np.where(np.diff(rates)*direction < -1e-10)[0]
            if not len(bad):break
            edges=np.delete(edges,int(bad[0])+1)
        return edges
    def fit(self,X,y):
        X=pd.DataFrame(X).copy(); y=pd.Series(y).reset_index(drop=True); X=X.reset_index(drop=True); self.edges_={}; self.maps_={}; self.iv_={}; tg=max((y==0).sum(),1); tb=max((y==1).sum(),1)
        for f in FEATURES:
            if f in NUMERIC:
                e=self._monotonic_edges(X[f],y); self.edges_[f]=e; key=pd.cut(X[f],e,include_lowest=True).astype(str)
            else: key=X[f].fillna("MISSING").astype(str)
            t=pd.DataFrame({"key":key,"y":y}).groupby("key").y.agg(["count","sum"]); t["good"]=t["count"]-t["sum"]; dg=(t.good+self.smoothing)/(tg+self.smoothing*len(t)); db=(t["sum"]+self.smoothing)/(tb+self.smoothing*len(t)); w=np.log(dg/db); self.maps_[f]=w.to_dict(); self.iv_[f]=float(((dg-db)*w).sum())
        return self
    def transform(self,X):
        X=pd.DataFrame(X).copy(); out=pd.DataFrame(index=X.index)
        for f in FEATURES:
            key=pd.cut(X[f],self.edges_[f],include_lowest=True).astype(str) if f in NUMERIC else X[f].fillna("MISSING").astype(str)
            out[f]=key.map(self.maps_[f]).fillna(0.0)
        return out
    def bin_label(self,f,value):
        if f in NUMERIC:return str(pd.cut(pd.Series([value]),self.edges_[f],include_lowest=True).iloc[0])
        return str(value)

class WOELogisticScorecard:
    def __init__(self,n_bins=8): self.woe=WOETransformer(n_bins); self.model=LogisticRegression(max_iter=1500,C=.6)
    def fit(self,X,y,sample_weight=None):
        z=self.woe.fit_transform(X,y); self.model.fit(z,y,sample_weight=sample_weight); return self
    def predict_proba(self,X): return self.model.predict_proba(self.woe.transform(X))
    def points_detail(self,X):
        X=pd.DataFrame(X).copy(); z=self.woe.transform(X); factor=50/np.log(2); rows=[]
        for i in range(len(X)):
            contributions=[]
            for j,f in enumerate(FEATURES):
                logodds=float(self.model.coef_[0,j]*z.iloc[i,j]); contributions.append({"feature":f,"bin":self.woe.bin_label(f,X.iloc[i][f]),"woe":float(z.iloc[i,j]),"logodds_contribution":logodds,"score_points":float(-factor*logodds)})
            rows.append(contributions)
        return rows
    def reason_codes(self,X,limit=3):
        names={"age":"AGE_PROFILE","monthly_income":"INCOME_PROFILE","employment_months":"EMPLOYMENT_HISTORY","bureau_score":"BUREAU_SCORE_PROFILE","existing_dti":"DEBT_BURDEN_PROFILE","requested_amount":"REQUESTED_AMOUNT_PROFILE","tenor_months":"TENOR_PROFILE","inquiries_6m":"RECENT_CREDIT_INQUIRIES","employment_type":"EMPLOYMENT_TYPE_PROFILE","home_ownership":"RESIDENCE_PROFILE","channel":"CHANNEL_PROFILE"}
        return [[names[x["feature"]] for x in sorted(r,key=lambda a:a["score_points"])[:limit] if x["score_points"]<0] or ["NO_MATERIAL_ADVERSE_FACTOR"] for r in self.points_detail(X)]

def ks_stat(y,p):
    f,t,_=roc_curve(y,p); return float(np.max(t-f))
def metrics(y,p):
    a=roc_auc_score(y,p); return {"auc":a,"gini":2*a-1,"ks":ks_stat(y,p),"brier":brier_score_loss(y,p),"mean_pd":float(np.mean(p)),"bad_rate":float(np.mean(y))}
