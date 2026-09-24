from pathlib import Path
import json
import unittest

ROOT = Path(__file__).resolve().parents[1]


class V6BankGradeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = json.loads((ROOT / "reports/validation_payload.json").read_text())

    def test_ifrs9_stages_reconcile(self):
        stages = self.p["v6_ifrs9_stage_summary"]
        self.assertEqual(sum(x["accounts"] for x in stages), 2400)
        self.assertTrue(all(x["weighted_ecl"] >= 0 for x in stages))
        self.assertTrue(all(0 <= x["coverage_ratio"] <= 1 for x in stages))

    def test_scenario_weights_and_order(self):
        s = self.p["v6_macro_scenarios"]
        self.assertAlmostEqual(sum(x["weight"] for x in s), 1.0, places=8)
        self.assertLess(s[0]["total_ecl"], s[1]["total_ecl"])
        self.assertLess(s[1]["total_ecl"], s[2]["total_ecl"])

    def test_delayed_labels_are_explicit(self):
        rows = self.p["v6_delayed_monitoring"]
        self.assertTrue(any(x["label_maturity_rate"] == 0 for x in rows))
        self.assertTrue(any(x["label_maturity_rate"] == 1 for x in rows))
        self.assertTrue(all(x["auc_when_mature"] is None for x in rows if x["label_maturity_rate"] == 0))

    def test_rollback_drill(self):
        gates = self.p["v6_deployment_gates"]
        self.assertEqual(gates[2]["status"], "ROLLBACK")
        self.assertEqual(gates[3]["status"], "BLOCKED")
        self.assertTrue(all(x["status"] == "PASS" for x in self.p["v6_rollback_drill"]))

    def test_lgd_ead_models(self):
        self.assertEqual(self.p["v6_lgd_validation"]["status"], "PASS")
        self.assertEqual(self.p["v6_ead_validation"]["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
