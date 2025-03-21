# sklearn_prg

A Python module implementing [Precision-Recall-Gain (PRG) curves](https://papers.nips.cc/paper_files/paper/2015/hash/33e8075e9970de0cfea955afd4644bb2-Abstract.html) and metrics compatible with scikit-learn.

## Why Precision-Recall-Gain?

Precision-Recall-Gain (PRG) curves improve upon traditional ROC and Precision-Recall curves in heavily imbalanced scenarios (few positives, many negatives). Use PRG when true negatives aren't valuable (e.g., information retrieval, fraud detection).

### Advantages:

- **Stable under imbalance:** Consistent evaluation regardless of class distribution.
- **Direct interpretation:** AUPRG directly relates to the expected F₁ score.
- **Improved model selection:** Avoids bias toward inflated metrics caused by imbalance.
- **Intuitive thresholding:** Convex hull easily identifies optimal thresholds for various Fβ scores.

## Installation

```bash
pip install sklearn_prg
```

## Usage

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_curve, precision_recall_curve, auc, average_precision_score)
from sklearn_prg.metrics import precision_recall_gain_curve, average_precision_recall_gain

# Generate imbalanced dataset
X, y = make_classification(n_samples=2000,
                           n_features=20,
                           weights=[0.9, 0.1],
                           flip_y=0.03,
                           class_sep=0.5,
                           random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# Train models
clf_rf = RandomForestClassifier(random_state=42).fit(X_train, y_train)
clf_lr = LogisticRegression(max_iter=1000, random_state=42).fit(X_train, y_train)
y_scores_rf = clf_rf.predict_proba(X_test)[:, 1]
y_scores_lr = clf_lr.predict_proba(X_test)[:, 1]

print(f"Average precision recall gain (Logistic Regression): {average_precision_recall_gain(y_test, y_scores_lr):.3f}")
print(f"Average precision recall gain (Random Forest): {average_precision_recall_gain(y_test, y_scores_rf):.3f}")

# ROC curve
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_scores_rf)
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_scores_lr)

# Standard PR curve
prec_rf, rec_rf, _ = precision_recall_curve(y_test, y_scores_rf)
prec_lr, rec_lr, _ = precision_recall_curve(y_test, y_scores_lr)
ap_rf = average_precision_score(y_test, y_scores_rf)
ap_lr = average_precision_score(y_test, y_scores_lr)

# PRG curve
pg_rf, rg_rf = precision_recall_gain_curve(y_test, y_scores_rf)
pg_lr, rg_lr = precision_recall_gain_curve(y_test, y_scores_lr)
auprg_rf = auc(rg_rf, pg_rf)
auprg_lr = auc(rg_lr, pg_lr)

fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# ----- ROC Curve -----
axs[0].plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC={auc(fpr_rf,tpr_rf):.3f})')
axs[0].plot(fpr_lr, tpr_lr, label=f'Logistic (AUC={auc(fpr_lr,tpr_lr):.3f})')
axs[0].plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Random Classifier')
axs[0].set_title('ROC Curves')
axs[0].set_xlabel('False Positive Rate')
axs[0].set_ylabel('True Positive Rate')
axs[0].legend()
axs[0].grid(True)
axs[0].set_aspect('equal', adjustable='box')

# ----- Precision-Recall Curve -----
prevalence = np.mean(y_test)
axs[1].plot(rec_rf, prec_rf, label=f'Random Forest (AP={ap_rf:.3f})')
axs[1].plot(rec_lr, prec_lr, label=f'Logistic (AP={ap_lr:.3f})')
axs[1].axhline(prevalence, linestyle='--', color='black', alpha=0.6, label='Random Classifier')
axs[1].set_title('Precision-Recall Curves')
axs[1].set_xlabel('Recall')
axs[1].set_ylabel('Precision')
axs[1].legend()
axs[1].grid(True)
axs[1].set_aspect('equal', adjustable='box')

# ----- Precision-Recall-Gain Curve -----
axs[2].plot(rg_rf, pg_rf, label=f'Random Forest (AUPRG={auprg_rf:.3f})')
axs[2].plot(rg_lr, pg_lr, label=f'Logistic (AUPRG={auprg_lr:.3f})')
axs[2].plot([1, 0], [0, 1], linestyle='-', color='black', alpha=0.6, label='Always Positive Classifier')
axs[2].set_xlim(0, 1)
axs[2].set_ylim(0, 1)
axs[2].set_xlabel('Recall Gain')
axs[2].set_ylabel('Precision Gain')
axs[2].set_title('Precision-Recall-Gain Curves')
axs[2].legend()
axs[2].grid(True)
axs[2].set_aspect('equal', adjustable='box')

plt.suptitle('ROC, PR, and PRG Curve Comparison', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
```

![A comparison of ROC, PR, and PRG curves](./images/prg_curves.png)

## Citation

If you use this package, please cite the original paper:

```bibtex
@inproceedings{NIPS2015_33e8075e,
 author = {Flach, Peter and Kull, Meelis},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {C. Cortes and N. Lawrence and D. Lee and M. Sugiyama and R. Garnett},
 pages = {},
 publisher = {Curran Associates, Inc.},
 title = {Precision-Recall-Gain Curves: PR Analysis Done Right},
 url = {https://proceedings.neurips.cc/paper_files/paper/2015/file/33e8075e9970de0cfea955afd4644bb2-Paper.pdf},
 volume = {28},
 year = {2015}
}
```
