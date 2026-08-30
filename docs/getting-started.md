# Getting Started

## Installation

```bash
pip install CausalEstimate
```

For local development:

```bash
git clone https://github.com/kirilklein/CausalEstimate.git
cd CausalEstimate
pip install -e .
```

## The input-data contract

Every estimator works on a plain pandas DataFrame. You tell the estimator which columns to use in its constructor; `compute_effect(df)` does the rest. CausalEstimate does **not** fit propensity or outcome models for you — you bring those predictions as columns:

| Column | Meaning | Needed by |
|--------|---------|-----------|
| treatment | Binary treatment assignment (0/1) | all estimators |
| outcome | Observed outcome | all estimators |
| ps | Propensity score, your estimate of P(treatment = 1 given covariates) | IPW, AIPW, TMLE, Matching |
| predicted_outcome | Predicted outcome under the *observed* treatment | TMLE |
| predicted_outcome_treated | Predicted outcome if treated | AIPW, TMLE |
| predicted_outcome_control | Predicted outcome if untreated | AIPW, TMLE |

Column names are arbitrary — pass them as `treatment_col=...`, `ps_col=...`, etc.

## First estimate

```python
import numpy as np
import pandas as pd
from CausalEstimate.estimators import IPW

np.random.seed(42)
n = 1000
ps = np.random.uniform(0, 1, n)          # true propensity for treatment
treatment = np.random.binomial(1, ps)    # actual treatment assignment
outcome = 2 + 0.5 * treatment + np.random.normal(0, 1, n)

df = pd.DataFrame({"ps": ps, "treatment": treatment, "outcome": outcome})

ipw_estimator = IPW(
    effect_type="ATE",
    treatment_col="treatment",
    outcome_col="outcome",
    ps_col="ps",
)

results = ipw_estimator.compute_effect(df)
print(results)
```

Output:

```python
{'effect': 0.5518, 'effect_1': 2.5260, 'effect_0': 1.9742}
```

`effect` is the estimated treatment effect; `effect_1` and `effect_0` are the mean potential outcomes under treatment and control. For standard errors and confidence intervals, see [Multiple Estimators & Bootstrap](user-guide/multi-estimator.md).
