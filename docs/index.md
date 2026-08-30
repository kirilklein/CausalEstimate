# CausalEstimate

**CausalEstimate** is a Python library for **causal inference**: it estimates treatment effects from observational data using doubly robust techniques such as **Targeted Maximum Likelihood Estimation (TMLE)**, alongside **propensity score**-based methods like inverse probability weighting (IPW) and matching.

## Why CausalEstimate?

Libraries like [DoWhy](https://github.com/py-why/dowhy), [EconML](https://github.com/py-why/EconML), and [causallib](https://github.com/BiomedSciAI/causallib) are powerful, but they couple effect estimation to their own model-fitting pipelines. CausalEstimate takes the opposite approach:

- **Bring your own predictions.** You fit propensity scores and outcome models however you like — scikit-learn, XGBoost, a deep model, or scores from an external system. CausalEstimate takes the resulting columns and estimates effects.
- **Pandas-native.** Input is a plain DataFrame with named columns; output is a plain dictionary. No wrappers, no custom data containers.
- **Lightweight.** A small dependency footprint (numpy, pandas, scipy, scikit-learn, statsmodels) and a focused scope: average effects with doubly robust estimators, matching, and bootstrap inference.

Reach for DoWhy/EconML instead when you want end-to-end pipelines, causal graphs, or heterogeneous (CATE) estimation.

## Quick example

```python
import numpy as np
import pandas as pd
from CausalEstimate.estimators import IPW

np.random.seed(42)
n = 1000
ps = np.random.uniform(0, 1, n)
treatment = np.random.binomial(1, ps)
outcome = 2 + 0.5 * treatment + np.random.normal(0, 1, n)
df = pd.DataFrame({"ps": ps, "treatment": treatment, "outcome": outcome})

ipw = IPW(effect_type="ATE", treatment_col="treatment", outcome_col="outcome", ps_col="ps")
print(ipw.compute_effect(df))
# {'effect': 0.5518, 'effect_1': 2.5260, 'effect_0': 1.9742}
```

## Where to go next

- [Getting Started](getting-started.md) — installation and the input-data contract
- [Estimators](user-guide/estimators.md) — IPW, AIPW, TMLE, Matching, and which effect types each supports
- [Multiple Estimators & Bootstrap](user-guide/multi-estimator.md) — run several estimators in one pass with confidence intervals
- [API Reference](api/estimators.md) — full signatures and docstrings
