import json
from pathlib import Path
m=json.loads((Path(__file__).resolve().parents[1]/"artifacts/model_metadata_v4.json").read_text())
checks={"AUC":m["auc"]>=.65,"KS":m["ks"]>=.25,"Brier":m["brier"]<.15,"Calibration gap":abs(m["mean_pd"]-m["bad_rate"])<.03}
print(checks)
if not all(checks.values()): raise SystemExit("Model approval gate failed")
