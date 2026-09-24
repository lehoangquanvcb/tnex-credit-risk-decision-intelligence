import json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class TestV9(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.p=json.loads((ROOT/"reports/validation_payload.json").read_text())
 def test_three_independent_models(self):
  self.assertEqual(len(self.p["v9_product_models"]),3);self.assertEqual(len({x["model_id"] for x in self.p["v9_product_models"]}),3)
 def test_allocation_reconciles(self):
  r=self.p["v9_recommended_allocation"];self.assertAlmostEqual(r["cash_loan_share"]+r["bnpl_share"]+r["business_loan_share"],1)
 def test_risk_appetite(self):self.assertEqual(self.p["v9_recommended_allocation"]["risk_appetite_status"],"PASS")
 def test_model_governance(self):self.assertTrue(all(x["owner"] and x["independent_validator"] and x["next_review"] for x in self.p["v9_model_inventory"]))
 def test_customer_outcomes(self):self.assertEqual(len(self.p["v9_customer_outcomes"]),9)
 def test_committee_pack(self):self.assertEqual(len(self.p["v9_committee_pack"]),4)
if __name__=="__main__":unittest.main()
