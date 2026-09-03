# Estimators

All estimators share the same pattern: configure columns and effect type in the constructor, then call `compute_effect(df)`.

## Supported effect types

| Estimator | ATE | ATT | RR | RRT | ARR |
|-----------|:---:|:---:|:--:|:---:|:---:|
| IPW       | ✓   | ✓   | ✓  | ✓   | ✓   |
| AIPW      | ✓   | ✓   | –  | –   | ✓   |
| TMLE      | ✓   | ✓   | ✓  | –   | ✓   |
| Matching  | ✓*  | –   | –  | –   | ✓*  |

ATE: average treatment effect · ATT: ATE on the treated · RR: risk ratio · RRT: risk ratio on the treated · ARR: absolute risk reduction.

\* With a caliper, the matched population is strictly neither the full nor the treated population; interpret accordingly.

## IPW — inverse probability weighting

Weights each observation by the inverse of its (estimated) probability of receiving the treatment it actually received. Only needs a propensity score column. Pass `stabilized=True` for stabilized weights.

```python
from CausalEstimate.estimators import IPW

ipw = IPW(effect_type="ATE", treatment_col="treatment", outcome_col="outcome", ps_col="ps")
result = ipw.compute_effect(df)
```

## AIPW — augmented IPW (doubly robust)

Combines the propensity score with outcome-model predictions. Consistent if *either* the propensity model or the outcome model is correctly specified.

```python
from CausalEstimate.estimators import AIPW

aipw = AIPW(
    effect_type="ATE",
    treatment_col="treatment",
    outcome_col="outcome",
    ps_col="ps",
    probas_t1_col="predicted_outcome_treated",
    probas_t0_col="predicted_outcome_control",
)
```

## TMLE — targeted maximum likelihood estimation

Doubly robust like AIPW, but updates the initial outcome predictions with a targeting step, which typically improves finite-sample behavior. Additionally requires predictions under the observed treatment.

```python
from CausalEstimate.estimators import TMLE

tmle = TMLE(
    effect_type="ATE",
    treatment_col="treatment",
    outcome_col="outcome",
    ps_col="ps",
    probas_col="predicted_outcome",
    probas_t1_col="predicted_outcome_treated",
    probas_t0_col="predicted_outcome_control",
)
```

## Matching

Estimates effects by comparing matched treated/control pairs on the propensity score. See [Matching](matching.md) for the standalone matching functions and their options.

## Weight clipping

`IPW`, `AIPW` and `TMLE` all accept `clip_percentile` (default `1`, no clipping) and `eps` (default `1e-9`). `clip_percentile=0.99` clips inverse-probability weights at their 99th percentile, which tames extreme propensity scores at the cost of some bias. Set the same value on every estimator in a `MultiEstimator` so their results stay comparable.

See the [API Reference](../api/estimators.md) for full signatures.
