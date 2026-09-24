import json, unittest
from pathlib import Path
from src.v8_digital_lending_os import live_decision

ROOT=Path(__file__).resolve().parents[1]

class TestV8(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.p=json.loads((ROOT/"reports/validation_payload.json").read_text())
    def test_product_models(self): self.assertEqual(len(self.p["v8_product_models"]),3)
    def test_optimizer_reconciles(self):
        r=self.p["v8_recommended_strategy"]
        self.assertAlmostEqual(r["contribution_vnd"],r["approved_exposure_vnd"]*.24-r["expected_loss_vnd"]-r["approved_exposure_vnd"]*.085-r["approved_accounts"]*115000-r["fraud_loss_vnd"],places=2)
    def test_payment_suppression(self): self.assertTrue(self.p["v8_summary"]["payment_contact_suppression_pass"])
    def test_live_cases_have_reason_codes(self): self.assertTrue(all(x["reason_codes"] and x["policy_version"]=="TNEX_SIM_2.0" for x in self.p["v8_live_cases"]))
    def test_network_actions(self): self.assertTrue(all(x["action"] in {"BLOCK_AND_INVESTIGATE","STEP_UP_VERIFICATION"} for x in self.p["v8_fraud_network"]))
    def test_live_decision_rejects_bad_product(self):
        with self.assertRaises(ValueError): live_decision({"product":"OTHER"})

if __name__=="__main__": unittest.main()
