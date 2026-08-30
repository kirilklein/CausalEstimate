# CausalEstimate

[![Unittests](https://github.com/kirilklein/CausalEstimate/actions/workflows/unittest.yml/badge.svg)](https://github.com/kirilklein/CausalEstimate/actions/workflows/unittest.yml)
[![Lint using flake8](https://github.com/kirilklein/CausalEstimate/actions/workflows/lint.yml/badge.svg)](https://github.com/kirilklein/CausalEstimate/actions/workflows/lint.yml)
[![Formatting using black](https://github.com/kirilklein/CausalEstimate/actions/workflows/format.yml/badge.svg)](https://github.com/kirilklein/CausalEstimate/actions/workflows/format.yml)
[![PyPI version](https://img.shields.io/pypi/v/CausalEstimate)](https://pypi.org/project/CausalEstimate/)
[![Python versions](https://img.shields.io/pypi/pyversions/CausalEstimate)](https://pypi.org/project/CausalEstimate/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-mkdocs%20material-blue)](https://kirilklein.github.io/CausalEstimate/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/kirilklein/CausalEstimate)

📖 **Documentation**: [kirilklein.github.io/CausalEstimate](https://kirilklein.github.io/CausalEstimate/)

Estimate average treatment effects from observational data with doubly robust estimators (TMLE, AIPW), inverse probability weighting, and matching — using propensity scores and outcome predictions you already have in a pandas DataFrame.

---

## Why CausalEstimate?

[DoWhy](https://github.com/py-why/dowhy), [EconML](https://github.com/py-why/EconML), and [causallib](https://github.com/BiomedSciAI/causallib) couple effect estimation to their own model-fitting pipelines. CausalEstimate does the opposite:

- **Bring your own predictions.** Fit propensity and outcome models with scikit-learn, XGBoost, a deep model, or an external system; CausalEstimate takes the resulting columns.
- **Pandas-native.** Input is a DataFrame with named columns; output is a plain dictionary.
- **Focused.** Average effects (ATE, ATT, risk ratios) with bootstrap inference and overlap diagnostics — no graphs, no CATE.

Reach for DoWhy/EconML when you want end-to-end pipelines, causal graphs, or heterogeneous effects.

---

## Installation

```bash
pip install CausalEstimate
```

---

## Quickstart

### Single estimator

Each estimator is configured with its column names and effect type, then called with `compute_effect(df)`.

```python
from CausalEstimate.datasets import load_binary_with_probas
from CausalEstimate.estimators import IPW

# Synthetic data with known ground truth. Columns "ps", "probas",
# "probas_t0", "probas_t1" stand in for your own model's predictions.
df, params = load_binary_with_probas(n_samples=5000, random_state=42, return_params=True)

ipw = IPW(effect_type="ATE", treatment_col="treatment", outcome_col="Y", ps_col="ps")
results = ipw.compute_effect(df)

print(f"IPW estimated effect: {results['effect']:.4f}")
print(f"True ATE: {params['true_ate']:.4f}")
```

```
IPW estimated effect: 0.1599
True ATE: 0.1644
```

`results["effect"]` is the estimated effect; `effect_1` / `effect_0` are the mean potential outcomes under treatment and control.

### Several estimators with bootstrap confidence intervals

`MultiEstimator` runs any set of estimators on the same data in one pass. `n_bootstraps > 1` adds `std_err` and percentile `CI95_lower` / `CI95_upper` to every result; `apply_common_support=True` trims units outside the propensity-score overlap region first.

```python
from CausalEstimate import MultiEstimator
from CausalEstimate.estimators import IPW, AIPW, TMLE

cols = dict(treatment_col="treatment", outcome_col="Y", ps_col="ps")
estimators = [
    IPW(effect_type="ATE", **cols),
    AIPW(effect_type="ATE", probas_t1_col="probas_t1", probas_t0_col="probas_t0", **cols),
    TMLE(effect_type="ATE", probas_col="probas", probas_t1_col="probas_t1", probas_t0_col="probas_t0", **cols),
]

results = MultiEstimator(estimators).compute_effects(df, n_bootstraps=100)
for name, r in results.items():
    print(f"{name:5s} effect={r['effect']:.4f}  95% CI=({r['CI95_lower']:.4f}, {r['CI95_upper']:.4f})")
```

```
IPW   effect=0.1614  95% CI=(0.1296, 0.1906)
AIPW  effect=0.1644  95% CI=(0.1412, 0.1878)
TMLE  effect=0.1649  95% CI=(0.1395, 0.1913)
```

Bootstrap resampling is not seeded, so your numbers will differ slightly.

---

## What's included

| Estimator | ATE | ATT | RR | RRT | ARR |
|-----------|:---:|:---:|:--:|:---:|:---:|
| IPW       | ✓   | ✓   | ✓  | ✓   | ✓   |
| AIPW      | ✓   | ✓   | –  | –   | ✓   |
| TMLE      | ✓   | ✓   | ✓  | –   | ✓   |
| Matching  | ✓*  | –   | –  | –   | ✓*  |

ATE: average treatment effect · ATT: ATE on the treated · RR: risk ratio · RRT: risk ratio on the treated · ARR: absolute risk reduction.
\* With a caliper the matched population is neither the full nor the treated population; interpret accordingly.

- **Diagnostics** (`CausalEstimate.diagnostics`) — covariate balance (`compute_balance_table`, `check_balance`; SMD with pooled unweighted SD, as in cobalt), positivity/overlap metrics, effective sample size and weight summaries.
- **Common-support filtering** (`CausalEstimate.filter_common_support`) — trim to the propensity-score overlap region.
- **Matching** (`CausalEstimate.estimators.Matching`) — greedy and optimal propensity-score matching with optional caliper.
- **Synthetic datasets** (`CausalEstimate.datasets`) — `load_binary`, `load_binary_with_probas`; return `true_ate`, `true_att`, `true_rr` for benchmarking.
- **Plotting** (`CausalEstimate.vis`) — propensity-score and outcome-probability distributions by treatment group.

---

## Documentation

- [Getting Started](https://kirilklein.github.io/CausalEstimate/getting-started/) — installation and the input-data contract
- [Estimators](https://kirilklein.github.io/CausalEstimate/user-guide/estimators/) — IPW, AIPW, TMLE, Matching
- [Multiple Estimators & Bootstrap](https://kirilklein.github.io/CausalEstimate/user-guide/multi-estimator/)
- [Matching](https://kirilklein.github.io/CausalEstimate/user-guide/matching/)
- [Diagnostics](https://kirilklein.github.io/CausalEstimate/user-guide/diagnostics/)
- [Plotting](https://kirilklein.github.io/CausalEstimate/user-guide/plotting/)
- [API Reference](https://kirilklein.github.io/CausalEstimate/api/estimators/)

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the dev setup and test workflow. Bug reports and feature requests are welcome as [issues](https://github.com/kirilklein/CausalEstimate/issues); questions can also go to [kikl@di.ku.dk](mailto:kikl@di.ku.dk).

## License

MIT — see [LICENSE](LICENSE).

## Citation

Use the "Cite this repository" button on GitHub (backed by [CITATION.cff](CITATION.cff)), or:

```bibtex
@software{causalestimate,
  author = {Klein, Kiril},
  title = {CausalEstimate: A Python Library for Causal Inference},
  year = {2024},
  url = {https://github.com/kirilklein/CausalEstimate},
  note = {GitHub repository}
}
```
