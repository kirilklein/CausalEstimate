# Multiple Estimators & Bootstrap

`MultiEstimator` runs several estimators on the **same** dataset in one pass — optionally with bootstrap standard errors and common-support filtering.

```python
from CausalEstimate import MultiEstimator
from CausalEstimate.estimators import IPW, AIPW, TMLE

ipw = IPW(effect_type="ATE", treatment_col="treatment", outcome_col="outcome", ps_col="ps")
aipw = AIPW(
    effect_type="ATE",
    treatment_col="treatment",
    outcome_col="outcome",
    ps_col="ps",
    probas_t1_col="predicted_outcome_treated",
    probas_t0_col="predicted_outcome_control",
)
tmle = TMLE(
    effect_type="ATE",
    treatment_col="treatment",
    outcome_col="outcome",
    ps_col="ps",
    probas_col="predicted_outcome",
    probas_t1_col="predicted_outcome_treated",
    probas_t0_col="predicted_outcome_control",
)

multi_estimator = MultiEstimator([ipw, aipw, tmle])

results = multi_estimator.compute_effects(
    df,
    n_bootstraps=50,                # > 1 triggers bootstrapping
    apply_common_support=True,
    common_support_threshold=0.05,
    return_bootstrap_samples=True,  # optionally return raw bootstrap estimates
)
print(results)
```

## Output format

`results` is a dictionary keyed by estimator class name (`"IPW"`, `"AIPW"`, `"TMLE"`). With bootstrapping (`n_bootstraps > 1`), each entry contains:

- `"effect"` — the mean effect across bootstrap samples
- `"std_err"` — the standard deviation of the bootstrap estimates
- `"CI95_lower"` / `"CI95_upper"` — the 95% confidence interval (percentile method)
- `"n_bootstraps"` — the number of bootstrap samples
- `"bootstrap_samples"` — the raw bootstrap estimates, if `return_bootstrap_samples=True`

Without bootstrapping (`n_bootstraps=1`), `"n_bootstraps"` is reported as 0 and the bootstrap summary keys (`"std_err"`, `"CI95_lower"`, `"CI95_upper"`) may be absent.

## Common-support filtering

`apply_common_support=True` restricts estimation to the region where treated and control propensity distributions overlap, trimming observations outside the shared support (controlled by `common_support_threshold`). This guards against extrapolation into regions where one group has no comparable units. The underlying function is also available directly as `CausalEstimate.filter_common_support` — see the [API Reference](../api/filtering.md).
