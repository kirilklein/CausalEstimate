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

---

**CausalEstimate** is a Python library designed for **causal inference**, providing a suite of methods to estimate treatment effects from observational data. It includes doubly robust techniques such as **Targeted Maximum Likelihood Estimation (TMLE)**, alongside **propensity score**-based methods like inverse probability weighting (IPW) and matching. The library is built for **flexibility** and **ease of use**, integrating seamlessly with pandas and supporting **bootstrap**-based standard error estimation and **multiple** estimators in one pass.

---

## Why CausalEstimate?

Libraries like [DoWhy](https://github.com/py-why/dowhy), [EconML](https://github.com/py-why/EconML), and [causallib](https://github.com/BiomedSciAI/causallib) are powerful, but they couple effect estimation to their own model-fitting pipelines. CausalEstimate takes the opposite approach:

- **Bring your own predictions.** You fit propensity scores and outcome models however you like — scikit-learn, XGBoost, a deep model, or scores from an external system. CausalEstimate takes the resulting columns and estimates effects.
- **Pandas-native.** Input is a plain DataFrame with named columns; output is a plain dictionary. No wrappers, no custom data containers.
- **Lightweight.** A small dependency footprint (numpy, pandas, scipy, scikit-learn, statsmodels) and a focused scope: average effects (ATE/ATT/RR) with doubly robust estimators, matching, and bootstrap inference.

Reach for DoWhy/EconML instead when you want end-to-end pipelines, causal graphs, or heterogeneous (CATE) estimation.

---

## Features

- **Causal inference methods** and the effect types each supports:

  | Estimator | ATE | ATT | RR | RRT | ARR |
  |-----------|:---:|:---:|:--:|:---:|:---:|
  | IPW       | ✓   | ✓   | ✓  | ✓   | ✓   |
  | AIPW      | ✓   | ✓   | –  | –   | ✓   |
  | TMLE      | ✓   | ✓   | ✓  | –   | ✓   |
  | Matching  | ✓*  | –   | –  | –   | ✓*  |

  ATE: average treatment effect · ATT: ATE on the treated · RR: risk ratio · RRT: risk ratio on the treated · ARR: absolute risk reduction.
  \* With a caliper, the matched population is strictly neither the full nor the treated population; interpret accordingly.

- **Bootstrap standard error estimation** and confidence intervals
- **Common-support filtering** and **matching** (greedy, optimal)
- **Plotting utilities** for distribution checks (e.g., propensity score overlap)
- **Diagnostics**: positivity/overlap metrics for propensity scores

---

## Installation

```bash
pip install CausalEstimate
```

Or for local development:

```bash
git clone https://github.com/kirilklein/CausalEstimate.git
cd CausalEstimate
pip install -e .
```

---

## Usage

### 1) Single Estimator Usage

You can import any estimator class (e.g., `IPW`, `AIPW`, `TMLE`) and call `compute_effect(df)` directly. Columns (treatment, outcome, propensity score) are passed to the estimator in its constructor.

```python
import numpy as np
import pandas as pd
from CausalEstimate.estimators import IPW

# Simulate data
np.random.seed(42)
n = 1000
ps = np.random.uniform(0, 1, n)          # true propensity for treatment
treatment = np.random.binomial(1, ps)    # actual treatment assignment
outcome = 2 + 0.5 * treatment + np.random.normal(0, 1, n)

df = pd.DataFrame({
    "ps": ps,
    "treatment": treatment,
    "outcome": outcome
})

# Create an IPW Estimator for ATE
ipw_estimator = IPW(
    effect_type="ATE",
    treatment_col="treatment",
    outcome_col="outcome",
    ps_col="ps",
    # optionally stabilized=True if you want stabilized IP weights
)

results = ipw_estimator.compute_effect(df)
print("IPW estimated effect:", results)
```

Output:

```python
{'effect': 0.5518, 'effect_1': 2.5260, 'effect_0': 1.9742}
```

`results` is a plain dictionary: `effect` is the estimated treatment effect, and `effect_1`/`effect_0` are the mean potential outcomes under treatment and control. When bootstrapping is applied (see [Multiple Estimators & Bootstrap](https://kirilklein.github.io/CausalEstimate/user-guide/multi-estimator/)), standard errors and confidence intervals are added.

---

## Documentation

Full documentation lives at **[kirilklein.github.io/CausalEstimate](https://kirilklein.github.io/CausalEstimate/)**:

- [Getting Started](https://kirilklein.github.io/CausalEstimate/getting-started/) — installation and the input-data contract
- [Estimators](https://kirilklein.github.io/CausalEstimate/user-guide/estimators/) — IPW, AIPW, TMLE, Matching
- [Multiple Estimators & Bootstrap](https://kirilklein.github.io/CausalEstimate/user-guide/multi-estimator/) — several estimators in one pass, with confidence intervals
- [Matching](https://kirilklein.github.io/CausalEstimate/user-guide/matching/) — optimal and greedy propensity-score matching
- [Diagnostics](https://kirilklein.github.io/CausalEstimate/user-guide/diagnostics/) — positivity/overlap metrics, effective sample size, weight summaries
- [Plotting](https://kirilklein.github.io/CausalEstimate/user-guide/plotting/) — propensity-score overlap checks
- [API Reference](https://kirilklein.github.io/CausalEstimate/api/estimators/) — full signatures and docstrings

---

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for details on setting up a dev environment, running tests, and contributing to this project.

---

## License

**CausalEstimate** is licensed under the MIT License. See [LICENSE](LICENSE) for more details.

---

## Contact

- **GitHub**: [kirilklein](https://github.com/kirilklein)
- **Email**: [kikl@di.ku.dk](mailto:kikl@di.ku.dk)

Please open issues or pull requests if you find any bugs or want to propose enhancements.

---

## Citation

If you use **CausalEstimate** in your research, please cite it via the "Cite this repository" button on GitHub (backed by [CITATION.cff](CITATION.cff)), or use the following BibTeX entry:

```bibtex
@software{causalestimate,
  author = {Klein, Kiril},
  title = {CausalEstimate: A Python Library for Causal Inference},
  year = {2024},
  url = {https://github.com/kirilklein/CausalEstimate},
  note = {GitHub repository}
}
```
