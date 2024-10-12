# CausalEstimate

[![Unittests](https://github.com/kirilklein/CausalEstimate/actions/workflows/unittest.yml/badge.svg)](https://github.com/kirilklein/CausalEstimate/actions/workflows/unittest.yml)
[![Lint using flake8](https://github.com/kirilklein/CausalEstimate/actions/workflows/lint.yml/badge.svg)](https://github.com/kirilklein/CausalEstimate/actions/workflows/lint.yml)
[![Formatting using black](https://github.com/kirilklein/CausalEstimate/actions/workflows/format.yml/badge.svg)](https://github.com/kirilklein/CausalEstimate/actions/workflows/format.yml)

---

**CausalEstimate** is a Python tool designed to produce causal estimates from propensity scores. It provides functionalities for matching, weighting, and other causal inference techniques, helping researchers and data scientists derive meaningful insights from observational data.

---

## Features

- Propensity score matching and weighting
- Tools for average treatment effect (ATE) estimation
- Easy integration with pandas DataFrames
- Bootstrap standard error estimation

---

## Installation

To install the required dependencies, run:

```sh
pip install -r requirements.txt
```

---

## Usage

### Example: Matching

Here is an example of how to use the matching functionality in a Jupyter notebook:

```python
import numpy as np
import pandas as pd
from CausalEstimate.matching import match_optimal

# Simulate data
ps = np.array([0.3, 0.90, 0.5, 0.34, 0.351, 0.32, 0.35, 0.81, 0.79, 0.77, 0.90, 0.6, 0.52, 0.55])
treated = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
ids = np.array([101, 102, 103, 103, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211])

df = pd.DataFrame({
    'PID': ids,
    'treatment': treated,
    'ps': ps
})

# Perform matching
result = match_optimal(df, n_controls=3, caliper=0.1)
print(result)
```

### Example: Using the Estimator

Here's an example of how to use the Estimator class to compute effects:

```python
import pandas as pd
import numpy as np
from CausalEstimate.interface.estimator import Estimator

# Simulate data
np.random.seed(42)
n = 1000
ps = np.random.uniform(0, 1, n)
treatment = np.random.binomial(1, ps)
outcome = 2 + 0.5*treatment + np.random.normal(0, 1, n)

df = pd.DataFrame({
    'treatment': treatment,
    'outcome': outcome,
    'ps': ps
})

# Create an Estimator object
estimator = Estimator(methods=['AIPW'], effect_type='ATE')

# Compute effects
results = estimator.compute_effect(
    df,
    treatment_col='treatment',
    outcome_col='outcome',
    ps_col='ps',
    bootstrap=True,
    n_bootstraps=100,
    method_args={},
    apply_common_support=False,
    common_support_threshold=0.1
)

print(results)
```

This example demonstrates how to:

1. Create an `Estimator` object with a specified method (AIPW in this case)
2. Use the `compute_effect` method to estimate the Average Treatment Effect (ATE)
3. Apply bootstrap for standard error estimation

---

## Development

### Running Tests

To run the unit tests, use the following command:

```sh
python -m unittest
```

### Linting

To lint the code using flake8, run:

```sh
pip install flake8
flake8 CausalEstimate tests
```

### Formatting

To format the code using black, run:

```sh
pip install black
black CausalEstimate tests
```

---

## License

This project is licensed under the MIT License.
