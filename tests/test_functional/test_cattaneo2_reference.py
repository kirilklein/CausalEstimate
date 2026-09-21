"""
External cross-check of the influence-curve standard errors, on real data.

The property-based tests in test_variance.py check the influence curves against
themselves -- mean-zero, bootstrap agreement, and an explicit restatement of
each EIF. All three are written against our own derivation, so a shared
misconception would pass them. This file pins the numbers against an
independent implementation instead: Stata's teffects estimators as ported to
statsmodels, which report GMM sandwich standard errors rather than influence
curves.

Data: cattaneo2 (Cattaneo 2010), maternal smoking and birthweight, n = 4642.
tests/data/cattaneo2.csv holds the seven columns these models need, taken from
the copy bundled with statsmodels. The outcome is the binary low-birthweight
indicator `lbweight`; `mage2` is exactly mage**2 and is derived here rather
than stored.

The nuisance models are statsmodels' own teffects specification, so both sides
see identical propensity scores and outcome regressions and the standard errors
are the only thing being compared. Note that the outcome model is a LINEAR
probability model: statsmodels' AIPW requires one and raises on a Logit, so
matching it is what keeps the comparison apples-to-apples.

Two kinds of assertion live here. The point estimates and our own standard
errors are pinned as a regression test, at rtol=1e-6 -- tight enough that any
real change lands far outside it (varying the nuisance specification moves
these in the fourth significant digit, a thousand times wider), but loose
enough to absorb platform differences in the iterative probit and GLM fits.
The agreement with the GMM standard errors is asserted at a much wider
tolerance, because the two are not the same estimator: GMM propagates the
uncertainty in the fitted nuisance models, while the influence curve treats
them as fixed, so exact equality is not expected in either direction.
"""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.discrete.discrete_model import Probit
from statsmodels.regression.linear_model import OLS

from CausalEstimate.estimators.functional.aipw import (
    compute_aipw_ate,
    compute_aipw_att,
)
from CausalEstimate.estimators.functional.ipw import compute_ipw_ate, compute_ipw_att
from CausalEstimate.estimators.functional.tmle import compute_tmle_ate
from CausalEstimate.estimators.functional.tmle_att import compute_tmle_att
from CausalEstimate.utils.constants import EFFECT, STD_ERR

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "cattaneo2.csv"

PS_FORMULA = "mbsmoke_ ~ mmarried_ + mage + mage2 + fbaby_ + medu"
OUTCOME_FORMULA = "lbweight ~ prenatal1_ + mmarried_ + mage + fbaby_"

# Produced by statsmodels 0.14.6, with the probit propensity model above and
# the linear outcome model above, via:
#
#   teff = TreatmentEffect(OLS.from_formula(OUTCOME_FORMULA, d), A,
#                          results_select=probit_result)
#   teff.ipw().summary_frame()                 -> ATE coef, std err
#   teff.ipw(effect_group=1).summary_frame()   -> ATT coef, std err
#   teff.aipw().summary_frame()                -> ATE std err
#   teff.ipw_ra(effect_group=1).summary_frame()-> ATT std err
#
# statsmodels reproduces Stata's teffects on this data, so these double as a
# check against Stata.
SM_IPW_ATE = 0.05201714
SM_IPW_ATT = 0.04749666
SM_IPW_ATE_GMM_SE = 0.01287185
SM_IPW_ATT_GMM_SE = 0.01209572
SM_AIPW_ATE_GMM_SE = 0.01305734
SM_IPWRA_ATT_GMM_SE = 0.01216344

# Our influence-curve standard errors on the same fits. Regression pins: any
# change to the variance layer must move these deliberately.
IC_SE = {
    "ipw/ATE": 0.01294992,
    "ipw/ATT": 0.01207411,
    # The AIPW pins carry the Hajek denominator contribution (the r term in
    # _compute_ic_mu); both moved down by well under a tenth of a percent when
    # it was added, because this outcome model leaves only a small mean
    # weighted residual. IPW and TMLE are unaffected by construction.
    "aipw/ATE": 0.01291868,
    "aipw/ATT": 0.01209017,
    "tmle/ATE": 0.01265818,
    "tmle/ATT": 0.01213402,
}

# Each IC standard error against the GMM standard error for the estimator
# statsmodels offers for that estimand.
GMM_COUNTERPART = {
    "ipw/ATE": SM_IPW_ATE_GMM_SE,
    "ipw/ATT": SM_IPW_ATT_GMM_SE,
    "aipw/ATE": SM_AIPW_ATE_GMM_SE,
    "aipw/ATT": SM_IPWRA_ATT_GMM_SE,
    "tmle/ATE": SM_AIPW_ATE_GMM_SE,
    "tmle/ATT": SM_IPWRA_ATT_GMM_SE,
}


class TestCattaneo2Reference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        d = pd.read_csv(DATA_PATH)
        d["mage2"] = d["mage"] ** 2

        cls.A = np.asarray(d["mbsmoke_"], dtype=float)
        cls.Y = np.asarray(d["lbweight"], dtype=float)
        cls.ps = np.asarray(
            Probit.from_formula(PS_FORMULA, d).fit(disp=0).predict(), dtype=float
        )
        # teffects fits the outcome regression separately per arm.
        treated = d["mbsmoke_"] == 1
        fit_1 = OLS.from_formula(OUTCOME_FORMULA, d[treated]).fit()
        fit_0 = OLS.from_formula(OUTCOME_FORMULA, d[~treated]).fit()
        cls.Y1_hat = np.asarray(fit_1.predict(d), dtype=float)
        cls.Y0_hat = np.asarray(fit_0.predict(d), dtype=float)

        cls.results = {
            "ipw/ATE": compute_ipw_ate(cls.A, cls.Y, cls.ps),
            "ipw/ATT": compute_ipw_att(cls.A, cls.Y, cls.ps),
            "aipw/ATE": compute_aipw_ate(cls.A, cls.Y, cls.ps, cls.Y0_hat, cls.Y1_hat),
            "aipw/ATT": compute_aipw_att(cls.A, cls.Y, cls.ps, cls.Y0_hat),
            "tmle/ATE": compute_tmle_ate(cls.A, cls.Y, cls.ps, cls.Y0_hat, cls.Y1_hat),
            "tmle/ATT": compute_tmle_att(cls.A, cls.Y, cls.ps, cls.Y0_hat, cls.Y1_hat),
        }

    def test_data_is_the_expected_extract(self):
        self.assertEqual(len(self.Y), 4642)
        self.assertEqual(int(self.A.sum()), 864)
        self.assertTrue(np.all(np.isin(self.Y, (0.0, 1.0))))

    def test_ipw_point_estimates_match_statsmodels(self):
        """
        The Hajek IPW estimators agree with statsmodels to every digit it
        reports, which is what makes the standard error comparison below a
        comparison of variance layers rather than of two different estimates.
        """
        np.testing.assert_allclose(
            self.results["ipw/ATE"][EFFECT], SM_IPW_ATE, rtol=1e-6
        )
        np.testing.assert_allclose(
            self.results["ipw/ATT"][EFFECT], SM_IPW_ATT, rtol=1e-6
        )

    def test_influence_curve_standard_errors_are_unchanged(self):
        """Regression pins for the variance layer on real data."""
        for label, expected in IC_SE.items():
            with self.subTest(label=label):
                np.testing.assert_allclose(
                    self.results[label][STD_ERR], expected, rtol=1e-6
                )

    def test_influence_curve_standard_errors_agree_with_gmm(self):
        """
        Within 5% of the GMM sandwich standard errors. The two estimators
        differ in whether the nuisance models are treated as estimated, so
        this is an agreement check, not an equality one; the observed spread
        is about 3% at its widest (TMLE ATE against AIPW).
        """
        for label, gmm in GMM_COUNTERPART.items():
            with self.subTest(label=label):
                ratio = self.results[label][STD_ERR] / gmm
                self.assertTrue(
                    0.95 <= ratio <= 1.05,
                    f"{label}: IC SE {self.results[label][STD_ERR]:.6f} vs GMM "
                    f"{gmm:.6f} (ratio {ratio:.4f})",
                )


if __name__ == "__main__":
    unittest.main()
