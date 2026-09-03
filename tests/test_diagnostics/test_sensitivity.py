import unittest

from CausalEstimate.diagnostics import compute_evalue


class TestComputeEvalue(unittest.TestCase):
    def test_published_example(self):
        # VanderWeele & Ding (2017): RR 3.9 (95% CI 1.8-8.5) -> E 7.26; CI E is 3.0 with the rounded bound 1.8
        res = compute_evalue(3.9, 1.8, 8.5)
        self.assertAlmostEqual(res["evalue"], 7.26, places=2)
        self.assertAlmostEqual(res["evalue_ci"], 3.0, places=9)

    def test_null_effect_gives_one(self):
        self.assertEqual(compute_evalue(1.0)["evalue"], 1.0)

    def test_no_ci_returns_none(self):
        self.assertIsNone(compute_evalue(2.0)["evalue_ci"])

    def test_protective_effect_uses_reciprocal(self):
        res = compute_evalue(0.5, 0.25, 0.8)
        self.assertAlmostEqual(res["evalue"], compute_evalue(2.0)["evalue"])
        self.assertAlmostEqual(res["evalue_ci"], compute_evalue(1.25)["evalue"])

    def test_ci_containing_null_gives_one(self):
        self.assertEqual(compute_evalue(1.5, 0.9, 2.5)["evalue_ci"], 1.0)
        self.assertEqual(compute_evalue(1.5, 1.0, 2.5)["evalue_ci"], 1.0)

    def test_risk_difference_scale(self):
        res = compute_evalue(0.1, 0.05, 0.15, scale="RD", baseline_risk=0.1)
        self.assertAlmostEqual(res["evalue"], compute_evalue(2.0)["evalue"])
        self.assertAlmostEqual(res["evalue_ci"], compute_evalue(1.5)["evalue"])

    def test_invalid_inputs_raise(self):
        with self.assertRaises(ValueError):
            compute_evalue(0.0)
        with self.assertRaises(ValueError):
            compute_evalue(2.0, ci_lower=1.5)
        with self.assertRaises(ValueError):
            compute_evalue(2.0, 3.0, 1.0)
        with self.assertRaises(ValueError):
            compute_evalue(2.0, scale="OR")
        with self.assertRaises(ValueError):
            compute_evalue(0.1, scale="RD")
        with self.assertRaises(ValueError):
            compute_evalue(-0.2, scale="RD", baseline_risk=0.1)


if __name__ == "__main__":
    unittest.main()
