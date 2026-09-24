from pathlib import Path
import json, unittest, joblib, pandas as pd

ROOT = Path(__file__).resolve().parents[1]

class ModelSmokeTest(unittest.TestCase):
    def test_artifacts_and_scoring(self):
        model = joblib.load(ROOT / "artifacts/retail_woe_scorecard_v4.joblib")
        row = pd.DataFrame([{"age":35,"monthly_income":20000000,"employment_months":60,"bureau_score":650,"existing_dti":.3,"requested_amount":50000000,"tenor_months":12,"inquiries_6m":1,"employment_type":"Salaried","home_ownership":"Owned","channel":"Mobile"}])
        p = model.predict_proba(row)[0,1]
        self.assertTrue(0 <= p <= 1)
        meta = json.loads((ROOT / "artifacts/model_metadata_v4.json").read_text())
        self.assertGreaterEqual(meta["auc"], .65)
        self.assertEqual(meta["model_id"], "RCS_PD_4.0")

if __name__ == "__main__":
    unittest.main()
