# Diagnostics

Positivity and weight diagnostics to report alongside IPW/TMLE estimates. Weighted estimates are only trustworthy when propensity scores stay away from 0 and 1 and no small set of observations dominates the weights — these functions quantify exactly that.

## Positivity / overlap

```python
from CausalEstimate.diagnostics import compute_positivity_metrics

# df has columns "ps" and "treatment"
metrics = compute_positivity_metrics(df, ps_col="ps", treatment_col="treatment")
print(metrics["prop_ps_extreme"])          # share of PS outside [eps, 1 - eps]
print(metrics["common_support_low"], metrics["common_support_high"])  # trimmed common support
print(metrics["flag_extreme_ps"])
```

The returned dictionary also includes group sizes, per-group extreme-PS shares, the share of observations outside the common support (overall and per group), and a Kolmogorov–Smirnov test comparing the treated and control propensity distributions (`ks_statistic`, `ks_p_value`).

## Weight diagnostics

Effective sample size (Kish) and weight summaries for the IPW weights the estimators use:

```python
from CausalEstimate.diagnostics import compute_ess, compute_weight_diagnostics

diag = compute_weight_diagnostics(df, ps_col="ps", treatment_col="treatment", weight_type="ATE")
print(diag["ess_total"], diag["ess_fraction_total"])   # ESS and ESS / n
print(diag["max_weight"], diag["weight_q99"])

# Or on your own weights (e.g. externally computed):
ess = compute_ess(weights)
```

A low ESS fraction means a few heavily weighted observations dominate the estimate — expect wide confidence intervals and sensitivity to those units. The total-sample keys (`ess_total`, `max_weight`, `weight_q95`, `weight_q99`, ...) are only included for `weight_type="ATE"`; ATT weights fix treated weights at 1, so per-group summaries are reported instead.

## Covariate balance

The standard pre/post-weighting SMD table. The denominator is the pooled unweighted SD, held fixed pre/post so both share a scale (Stuart 2010; cobalt's default):

```python
from CausalEstimate.diagnostics import compute_balance_table, check_balance
from CausalEstimate.vis.plotting import plot_love

table = compute_balance_table(df, covariate_cols=["age", "bmi"], ps_col="ps", treatment_col="treatment")
print(table)                    # means, smd_unweighted, smd_weighted, balanced per covariate
print(check_balance(table))     # max_smd_weighted, n_unbalanced, n_undefined, prop_unbalanced, balanced

fig, ax = plot_love(table)      # requires the plotting extra
```

Or run every diagnostic in one call with `run_diagnostics` (exported from the top-level package) — see the [README](https://github.com/kirilklein/CausalEstimate#readme) example and the [diagnostics notebook](https://github.com/kirilklein/CausalEstimate/blob/main/examples/diagnostics_example.ipynb).

See the [API Reference](../api/diagnostics.md) for full signatures, and [Plotting](plotting.md) for visual overlap checks.
