# Pricing Azerbaijani Real Estate from Scratch

**Decision Trees and Pegasos SVMs on bina.az Listings**
MLE-AI-201 Machine Learning, AI Academy (Cohort II): Final Project

Authors: Nurana Aliyeva, Sanan Garibli, Humbat Jamalov, Farah Nematzada, Elsen Shukurov

We predict real-estate prices from 100,775 [bina.az](https://bina.az) sale listings using two models written **from scratch in NumPy**:

- a **CART-style decision tree** (Gini, entropy and MSE criteria, vectorised split search, four stopping rules);
- a **soft-margin SVM trained with Pegasos** (sub-gradient descent), with **random Fourier features** as a non-linear extension.

Both are benchmarked against matched scikit-learn models (5-fold CV plus one test evaluation). scikit-learn is used for the baselines only.

## Tasks

| Task | Target | Metric |
|------|--------|--------|
| **A: Regression** | `log(price)` | RMSE, MAE, R² (and AZN errors with Duan's smearing correction) |
| **B: Classification** | Price tier: *premium* if price > training median (210,000 AZN), else *standard* | Precision, recall, F1, ROC-AUC |

## Headline results

Test set, one evaluation (seed 42, frozen split):

| Model | Task A: RMSE (log) | Task A: R² | Task B: F1 |
|-------|-------------------|-----------|-----------|
| **Our decision tree** | **0.332** | **0.802** | **0.929** |
| scikit-learn tree | 0.332 | 0.802 | 0.930 |
| Our Pegasos SVM | n/a | n/a | 0.843 |
| scikit-learn SGD (hinge) | n/a | n/a | 0.839 |
| scikit-learn LinearSVC | n/a | n/a | 0.842 |

5-fold CV on the training split: our tree matches scikit-learn within one standard deviation (RMSE 0.3377 vs. 0.3369; F1 0.9166 vs. 0.9162). Our SVM matches the linear scikit-learn baselines (F1 0.843 vs. 0.840 and 0.842).

Key findings:

- The tree is the best model on both tasks. A linear SVM cannot express the interactions (area × district × category) that drive price, and random Fourier features (D = 500) did not help.
- Errors concentrate at the price extremes (cheapest and most expensive quintiles) and near the 210,000 AZN tier threshold.
- Our implementations are roughly 9 to 23 times slower to fit than scikit-learn's compiled code, with equivalent accuracy.

See the report for the full tables, tuning sweeps and error analysis.

## Data

- Source: Kaggle **Bina.az Sale Project** (`sehriyarmemmedli/binaaz-sale-project`), file `house_sale.csv` (100,775 rows × 51 columns).
- The raw data is **not committed**. Download it and place `house_sale.csv` in `data/` (see `data/README.md`).

Cleaning pipeline (`data_prep.clean`, every decision is logged and printed by `run_all`):

- **Duplicates:** 36,321 repeat scrapes removed (same URL, latest kept).
- **Types and currency:** prices converted to AZN; area strings (`85 m²`, `6 sot`) and floors (`3 / 9`) parsed into numbers.
- **Outliers:** fixed domain limits (prices outside 10,000 to 5,000,000 AZN, impossible or per-category outlier areas); coordinates outside Azerbaijan set to missing. Limits are not data quantiles, so they cannot leak test information.
- **Leakage:** `unit_price` and `total_price` (functions of the target), identifiers, free text and scrape metadata are removed.
- **Missing values:** medians computed on training rows only, plus a `_missing` flag per imputed column.

Result: 63,822 listings, 72 features, a seeded stratified **70/15/15** split (44,676 / 9,573 / 9,573) frozen in `data/split_indices.npz`. Classes are balanced (49.3% premium in every split).

## Repo layout

```
.
├── README.md
├── requirements.txt          # pinned dependencies
├── .gitignore
├── contribution_report.md    # who did what
├── data/                     # dataset goes here (not committed); split_indices.npz
├── src/                      # all from-scratch code
│   ├── data_prep.py          #   loading, cleaning, split, tier label
│   ├── decision_tree.py      #   DecisionTree (Gini / entropy / MSE)
│   ├── svm.py                #   PegasosSVM, RFFPegasosSVM
│   ├── evaluate.py           #   metrics + comparison helpers
│   └── run_all.py            #   ONE command reproduces every number and figure
├── tests/
│   └── test_decision_tree.py # node-by-node comparison with scikit-learn
├── report/                   # IEEE two-column report
│   ├── report.tex
│   └── figures/
└── presentation/             # slides PDF
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# put house_sale.csv in data/ (see data/README.md), then:
python -m src.run_all
```

`python -m src.run_all` reproduces every headline number and figure in the report from a clean checkout. All random steps use seed 42 and the split is frozen, so reruns give identical numbers.

## Implementation notes

**Decision tree** (`src/decision_tree.py`)

- One `DecisionTree` class for classification (Gini, entropy) and regression (MSE).
- Vectorised split search: each feature is sorted once and cumulative sums score all candidate thresholds in a single pass, `O(n d log n)` per node instead of `O(n² d)`.
- Four stopping rules: `max_depth`, `min_samples_split`, `min_samples_leaf`, `min_impurity_decrease` (scikit-learn's weighted definition).
- Explicit stack instead of recursion, so very deep trees cannot hit Python's recursion limit.
- Reports class probabilities (`predict_proba`) and impurity-based feature importances; validates inputs (NaN/inf, column count).
- Ties between equal-gain features go to the lowest feature index (scikit-learn picks randomly), which explains small differences in tree structure (99.1% identical test predictions for Task B, 94.2% for Task A).

**Pegasos SVM** (`src/svm.py`)

- Mini-batch sub-gradient steps on the regularised hinge loss, with three learning-rate schedules (Pegasos `1/(λt)`, `η₀/√t`, constant), optional projection, and tail averaging of the last epoch.
- Logs the objective every epoch; reports `‖w‖`, the margin width `2/‖w‖` and the support vectors.
- The bias is treated as a weight on a constant-1 feature (a small departure from the textbook objective). Features are standardised with training statistics.
- `RFFPegasosSVM` adds random Fourier features (`D = 500`) in front of the unchanged linear solver to approximate an RBF kernel.

**Final configurations**

- Task A tree: `min_samples_split=75`, no depth limit.
- Task B tree: Gini, `max_depth=20`, `min_impurity_decrease=2e-5`.
- SVM: `λ = 1e-3`.

All hyperparameters are chosen on validation data (or CV on the training split); the test set is evaluated once.

## Tests

```bash
python -m pytest -q
```

The tests compare our tree node by node (feature, threshold, samples, value) with scikit-learn on a toy set and on seeded random data for all three criteria. Each example is first checked to give the same scikit-learn tree across ten random seeds, so the expected tree is fully determined. Further tests cover each stopping rule, `predict_proba` and input validation.

## Deliverables

- **Report:** `report/report.tex` (IEEE-style, self-contained, no `IEEEtran.cls`).
- **Slides:** `presentation/`
- **Contributions:** `contribution_report.md`
- **Release tag:** `v1.0-final`

## Limitations

Asking prices from a single scrape (not transaction prices); AZN errors are dominated by a few multi-million listings (test RMSE 253,669 AZN, MAE 79,550 AZN), so the log scale is the fairer summary; land plots are the hardest listings to price; the CV search for the final tree settings is not part of `run_all`.

## Acknowledgements

Data: Kaggle *Bina.az Sale Project*. Libraries: NumPy, pandas, Matplotlib, scikit-learn (baselines only). AI assistance: Claude (Anthropic) was used to test initial versions drafted according to each member's role; all code was reviewed in pull requests and checked against scikit-learn.
