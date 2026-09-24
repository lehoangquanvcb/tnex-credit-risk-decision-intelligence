from pathlib import Path
import json
import unittest

ROOT = Path(__file__).resolve().parents[1]


class V5DecisioningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = json.loads((ROOT / "reports/validation_payload.json").read_text())

    def test_walk_forward_has_four_folds(self):
        rows = self.payload["v5_walk_forward"]
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(r["auc"] >= 0.65 for r in rows))

    def test_strategy_is_economically_reconciled(self):
        for row in self.payload["v5_strategy_grid"]:
            self.assertGreaterEqual(row["expected_loss"], 0)
            self.assertGreaterEqual(row["total_ead"], 0)
            self.assertLessEqual(row["approval_rate"] + row["review_rate"], 1.000001)

    def test_stress_is_ordered(self):
        rows = self.payload["v5_stress_testing"]
        self.assertEqual([r["scenario"] for r in rows], ["Base", "Downturn", "Severe"])
        self.assertTrue(rows[0]["portfolio_pd"] < rows[1]["portfolio_pd"] < rows[2]["portfolio_pd"])
        self.assertTrue(rows[0]["expected_profit"] > rows[1]["expected_profit"] > rows[2]["expected_profit"])

    def test_challenger_rule_is_explicit(self):
        summary = self.payload["v5_summary"]
        self.assertIn(summary["challenger_decision"], ["RETAIN CHAMPION", "VALIDATE CHALLENGER"])
        self.assertEqual(summary["build"], "V5.0")


if __name__ == "__main__":
    unittest.main()
