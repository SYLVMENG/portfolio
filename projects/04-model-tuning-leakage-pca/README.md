# Leakage-safe model tuning and PCA trade-offs

Two common modelling mistakes, measured instead of just described.

![Tuned fit, and bias of the reported test error over 50 seeds](figures/fig_1.png)

## 1. Tuning on the test set
A polynomial ridge regression is tuned two ways: one picks hyperparameters using the test set, the other uses cross-validation on training data only. Repeating the experiment over 50 random seeds shows the leaked setup reports a test error that is on average **0.045 lower** than it should be (SE 0.013). The single-split numbers alone can't show this.

## 2. Does PCA help?
- On Iris, logistic regression on the first 2 principal components keeps the same weighted F1 (0.97) as all four features; one component drops to 0.93.
- For a noisy polynomial regression, adding PCA before ridge does not beat the tuned baseline: compared fold by fold, the difference (0.0045 in MSE) is within noise, and the baseline wins 3 of 5 folds.

Code is available on request.

**Tools:** Python, scikit-learn (Pipeline, GridSearchCV, PCA), NumPy, matplotlib.
