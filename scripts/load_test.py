import time,sys,joblib,pandas as pd,numpy as np
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.modeling import FEATURES
root=Path(__file__).resolve().parents[1];model=joblib.load(root/"artifacts/retail_woe_scorecard_v4.joblib");data=pd.read_csv(root/"data/development_sample.csv").tail(1000);lat=[]
for _,row in data.iterrows():
    start=time.perf_counter();model.predict_proba(pd.DataFrame([row[FEATURES]]));lat.append((time.perf_counter()-start)*1000)
p95=float(np.percentile(lat,95));print({"requests":len(lat),"p95_ms":round(p95,2),"mean_ms":round(float(np.mean(lat)),2)})
if p95>100:raise SystemExit("p95 latency gate failed")
