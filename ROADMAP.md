# Roadmap

Ideas borrowed from [zEpid](https://github.com/pzivich/zEpid) (unmaintained since 2023) that fit
CausalEstimate's scope: bring-your-own-predictions, pandas-native, average effects.
Take the estimands and diagnostics, not the string-formula API.

Status: `[ ]` open · `[~]` in progress · `[x]` done (link the PR).

## Plots

- [x] Love plot of covariate balance (#122)
- [x] Propensity-score boxplots before/after IPW weighting (`plot_ps_boxplot`, #134)
- [x] Zipper plot of CI coverage across simulation replicates (`plot_zipper`, #134)

## Diagnostics and weights

- [ ] Explicit weight truncation (symmetric / percentile bounds) reported in `compute_weight_diagnostics`
- [ ] Censoring / missingness weights: accept a censoring-probability column and fold it into IPW/AIPW weights

## Estimators

- [ ] Cross-fit AIPW/TMLE: fold-id column plus per-fold nuisance predictions
- [ ] Generalizability / transportability (IPSW, g-transport, AIPSW) from a sampling-probability column
- [ ] Rubin's rules for pooling effects across multiply imputed datasets

## Sensitivity analysis

- [ ] E-value for unmeasured confounding
- [ ] Monte Carlo bias analysis for RR (trapezoidal prior)

## Project

- [x] Logo at the top of the README (#135)

## Out of scope

Time-varying g-formula, structural nested models, SuperLearner, DAG tooling, and the 2x2
epidemiology calculators: these duplicate DoWhy/statsmodels or require fitting models inside the package.
