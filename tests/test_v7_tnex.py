import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestV7TNEX(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = json.loads((ROOT / "reports/validation_payload.json").read_text())

    def test_product_limits(self):
        limits = {x["product"]: x["max_limit_vnd"] for x in self.p["v7_product_config"]}
        self.assertEqual(limits, {"CASH_LOAN": 70_000_000, "BNPL": 25_000_000, "BUSINESS_LOAN": 100_000_000})

    def test_dual_decision_fields(self):
        row = self.p["v7_fraud_credit"][0]
        self.assertTrue({"decision_pd", "fraud_score", "decision", "primary_reason"} <= row.keys())

    def test_realtime_sla(self):
        self.assertLess(self.p["v7_summary"]["p95_end_to_end_ms"], 5000)

    def test_funnel_is_monotonic(self):
        n = [x["customers"] for x in self.p["v7_funnel"]]
        self.assertEqual(n, sorted(n, reverse=True))

    def test_feature_governance(self):
        self.assertTrue(all(x["owner"] and x["fallback"] for x in self.p["v7_feature_governance"]))

    def test_payment_suppression(self):
        c = {x["control"]: x for x in self.p["v7_collections"]}
        self.assertEqual(c["Reminder suppression"]["exceptions"], 0)


if __name__ == "__main__": unittest.main()
