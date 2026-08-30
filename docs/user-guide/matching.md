# Matching

CausalEstimate supports both **optimal** and **greedy** (a.k.a. eager) propensity-score matching.

```python
import pandas as pd
from CausalEstimate.matching import match_optimal, match_eager

df = pd.DataFrame({
    "PID": [101, 102, 103, 202, 203, 204],
    "treatment": [1, 1, 1, 0, 0, 0],
    "ps": [0.30, 0.35, 0.90, 0.31, 0.34, 0.85],
})

# Optimal matching (caliper=0.05, 1 control per treated)
matched_optimal = match_optimal(
    df, n_controls=1, caliper=0.05,
    treatment_col="treatment", ps_col="ps", pid_col="PID",
)

# Eager (greedy) matching
matched_eager = match_eager(
    df, caliper=0.05,
    treatment_col="treatment", ps_col="ps", pid_col="PID",
)
```

Both functions return a DataFrame of matched pairs (or sets) with columns like `[treated_pid, control_pid, distance]`.

**Optimal** matching minimizes the total propensity-score distance across all pairs; **eager** matching assigns the nearest available control to each treated unit in turn (faster, but can be globally suboptimal).

!!! note "Interpreting matched estimates"
    With a caliper, unmatched units are dropped, so the matched population is strictly neither the full population (ATE) nor the treated population (ATT). The `Matching` estimator reports it as ATE with this caveat.

See the [API Reference](../api/matching.md) for full signatures.
