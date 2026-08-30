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

See [this notebook](https://github.com/kirilklein/CausalEstimate/blob/main/examples/plot_examples.ipynb) for rendered examples, and the [API Reference](../api/plotting.md) for full signatures.
