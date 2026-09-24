import json,unittest
from pathlib import Path
import joblib,numpy as np,pandas as pd
from src.modeling import FEATURES
from src.policy import decide,load_policy
ROOT=Path(__file__).resolve().parents[1]
class HardeningControlsTest(unittest.TestCase):
    def test_scorecard_reconstruction(self):
        model=joblib.load(ROOT/"artifacts/retail_woe_scorecard_v4.joblib");data=pd.read_csv(ROOT/"data/development_sample.csv").tail(100);z=model.woe.transform(data[FEATURES]);logit=model.model.intercept_[0]+z.to_numpy()@model.model.coef_[0];manual=1/(1+np.exp(-logit));api=model.predict_proba(data[FEATURES])[:,1];self.assertLess(np.max(np.abs(manual-api)),1e-6)
    def test_policy_version_and_boundaries(self):
        p=load_policy();self.assertEqual(p["policy_version"],"RETAIL_POLICY_1.0");self.assertEqual(decide(.04,20_000_000,50_000_000,.3,12,p)["risk_band"],"A");self.assertEqual(decide(.20,20_000_000,50_000_000,.3,12,p)["decision"],"DECLINE");self.assertIn("MAXIMUM_DTI",decide(.04,20_000_000,50_000_000,.8,12,p)["policy_failures"])
    def test_synchronized_payload(self):
        p=json.loads((ROOT/"reports/validation_payload.json").read_text());meta=p["v41_metadata"]
        for key in ["v41_calibration","v41_monitoring","v41_fairness","v41_reject_inference"]:
            self.assertTrue(p[key]);self.assertTrue(all(x["model_id"]==meta["model_id"] and x["dataset_id"]==meta["dataset_id"] for x in p[key]))
        self.assertEqual(p["reconstruction"]["status"],"PASS")
if __name__=="__main__":unittest.main()
