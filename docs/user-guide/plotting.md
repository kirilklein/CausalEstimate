# Plotting

Plotting utilities help you check distributions of propensity scores and predicted outcome probabilities across treatment and control groups — the first sanity check before trusting any estimate.

Requires the plotting extra:

```bash
pip install "CausalEstimate[plotting]"
```

```python
import matplotlib.pyplot as plt
from CausalEstimate.vis.plotting import plot_propensity_score_dist, plot_outcome_proba_dist

# df has columns "ps", "treatment", and "predicted_outcome"
fig, ax = plot_propensity_score_dist(df, ps_col="ps", treatment_col="treatment")
plt.show()

fig, ax = plot_outcome_proba_dist(df, outcome_proba_col="predicted_outcome", treatment_col="treatment")
plt.show()
```

Good overlap between the treated and control propensity distributions supports the positivity assumption; clear separation is a warning sign — consider [common-support filtering](multi-estimator.md#common-support-filtering) or matching.

Two diagnostics-oriented plots complement the [Diagnostics](diagnostics.md) module: a Love plot of covariate balance and the IPW weight distribution.

```python
from CausalEstimate.diagnostics import compute_balance_table
from CausalEstimate.vis.plotting import plot_love, plot_weight_dist

table = compute_balance_table(df, covariate_cols=["age", "bmi"], ps_col="ps", treatment_col="treatment")
fig, ax = plot_love(table)
fig, ax = plot_weight_dist(df, ps_col="ps", treatment_col="treatment", weight_type="ATE")
```

See [this notebook](https://github.com/kirilklein/CausalEstimate/blob/main/examples/plot_examples.ipynb) and the [diagnostics notebook](https://github.com/kirilklein/CausalEstimate/blob/main/examples/diagnostics_example.ipynb) for rendered examples, and the [API Reference](../api/plotting.md) for full signatures.
