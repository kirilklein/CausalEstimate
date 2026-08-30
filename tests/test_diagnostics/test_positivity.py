import unittest

import numpy as np
import pandas as pd

from CausalEstimate.diagnostics import compute_positivity_metrics
from CausalEstimate.filter.propensity import get_common_support_range
from CausalEstimate.utils.constants import PS_COL, TREATMENT_COL
from tests.helpers.setup import TestEffectBase


class TestPositivityOnSimulatedData(TestEffectBase):
    def test_well_overlapping_data_has_no_violations(self):
        metrics = compute_positivity_metrics(self.data)
        self.assertEqual(metrics["n_total"], self.n)
        self.assertEqual(
            metrics["n_treated"] + metrics["n_control"], metrics["n_total"]
        )
        self.assertEqual(metrics["prop_ps_extreme"], 0.0)
        self.assertFalse(metrics["flag_extreme_ps"])
        self.assertGreater(metrics["ks_statistic"], 0.0)

    def test_support_range_matches_filter_module(self):
        metrics = compute_positivity_metrics(self.data, common_support_threshold=0.05)
        low, high = get_common_support_range(self.data, TREATMENT_COL, PS_COL, 0.05)
        self.assertEqual(metrics["common_support_low"], low)
        self.assertEqual(metrics["common_support_high"], high)


class TestPositivityExactProportions(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                TREATMENT_COL: [1, 1, 1, 1, 0, 0, 0, 0],
                PS_COL: [0.005, 0.4, 0.6, 0.995, 0.2, 0.3, 0.7, 0.8],
            }
        )

    def test_extreme_proportions(self):
        metrics = compute_positivity_metrics(self.df, eps=0.01)
        self.assertEqual(metrics["prop_ps_below_eps"], 1 / 8)
        self.assertEqual(metrics["prop_ps_above_1_minus_eps"], 1 / 8)
        self.assertEqual(metrics["prop_ps_extreme"], 2 / 8)
        self.assertEqual(metrics["prop_treated_ps_extreme"], 2 / 4)
        self.assertEqual(metrics["prop_control_ps_extreme"], 0.0)
        self.assertTrue(metrics["flag_extreme_ps"])

    def test_wider_eps_catches_more(self):
        metrics = compute_positivity_metrics(self.df, eps=0.25)
        self.assertEqual(metrics["prop_ps_extreme"], 4 / 8)

    def test_outside_support_proportions(self):
        metrics = compute_positivity_metrics(self.df, common_support_threshold=0.0)
        self.assertEqual(metrics["common_support_low"], 0.2)
        self.assertEqual(metrics["common_support_high"], 0.8)
        self.assertEqual(metrics["prop_treated_outside_support"], 2 / 4)
        self.assertEqual(metrics["prop_control_outside_support"], 0.0)
        self.assertEqual(metrics["prop_outside_support"], 2 / 8)

    def test_custom_column_names(self):
        renamed = self.df.rename(columns={TREATMENT_COL: "arm", PS_COL: "score"})
        metrics = compute_positivity_metrics(
            renamed, ps_col="score", treatment_col="arm"
        )
        self.assertEqual(metrics["prop_ps_extreme"], 2 / 8)

    def test_single_arm_raises(self):
        treated_only = self.df[self.df[TREATMENT_COL] == 1]
        with self.assertRaises(ValueError):
            compute_positivity_metrics(treated_only)

    def test_invalid_inputs_raise(self):
        nan_ps = self.df.assign(**{PS_COL: [0.5, np.nan] + [0.5] * 6})
        with self.assertRaises(ValueError):
            compute_positivity_metrics(nan_ps)
        ps_out_of_range = self.df.assign(**{PS_COL: [1.5] + [0.5] * 7})
        with self.assertRaises(ValueError):
            compute_positivity_metrics(ps_out_of_range)
        nonbinary_treatment = self.df.assign(
            **{TREATMENT_COL: [2, 1, 1, 1, 0, 0, 0, 0]}
        )
        with self.assertRaises(ValueError):
            compute_positivity_metrics(nonbinary_treatment)
        with self.assertRaises(ValueError):
            compute_positivity_metrics(self.df, eps=0.0)
        with self.assertRaises(ValueError):
            compute_positivity_metrics(self.df, common_support_threshold=0.5)
        with self.assertRaises(ValueError):
            compute_positivity_metrics(self.df.drop(columns=[PS_COL]))

    def test_result_types_are_plain_python(self):
        metrics = compute_positivity_metrics(self.df)
        self.assertIsInstance(metrics["n_total"], int)
        self.assertIsInstance(metrics["prop_ps_extreme"], float)
        self.assertIsInstance(metrics["flag_extreme_ps"], bool)
        self.assertNotIsInstance(metrics["flag_extreme_ps"], np.bool_)


if __name__ == "__main__":
    unittest.main()
