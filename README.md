# Decision Tree & Random Forest Classification on the Wine Dataset

A comprehensive, end-to-end implementation and visual companion guide comparing a **Single Decision Tree** against a **Random Forest Classifier** on the UCI Wine dataset (`178 samples`, `13 chemical features`, `3 cultivar classes`) using `scikit-learn`.

![Decision Tree vs. Random Forest Infographic](./random%20and%20decision%20tree.jpg)

## 📌 Project Overview

Tree-based models offer strong interpretability on tabular data, yet single unpruned trees suffer from high variance and memorization. This project demonstrates how depth constraints, bootstrap aggregation (bagging), and random feature subsetting resolve overfitting while exposing noisy features:

* **Single Decision Tree & Overfitting:** Demonstrates how an unconstrained tree memorizes training data (`100%` train accuracy vs. `~88.9%` test accuracy) and how tuning `max_depth` improves generalization.
* **Random Forest (Bagging & Randomness):** Combines multiple decorrelated trees trained on bootstrap samples with random feature subsets ($m = \sqrt{p} \approx 3$ features per split), using **Out-of-Bag (OOB)** evaluation (`~0.979` OOB score) as a built-in validation check.
* **Model Benchmarking & Feature Importance:** Evaluates models via **Repeated Stratified Cross-Validation** (`5-fold, 10-repeat`) and compares impurity-based vs. permutation-based feature importance using a synthetic `random_noise` column test.

## 📂 Repository Contents

| File | Description |
| :--- | :--- |
| [`wine_tree_forest.py`](./wine_tree_forest.py) | End-to-end Python script covering data splitting, depth tuning, Random Forest OOB validation, repeated stratified CV, and feature importance diagnostics. |
| [`Decision_Tree_Random_Forest_Wine.pdf`](./Decision_Tree_Random_Forest_Wine.pdf) | Visual companion guide covering splitting criteria, variance reduction, hyperparameter tuning, and `scikit-learn` walkthroughs. |
| [`random and decision tree.jpg`](./random%20and%20decision%20tree.jpg) | Visual summary infographic of the workflow, experimental benchmarks, and key takeaways. |

## 📊 Key Experimental Findings

* **Depth Tuning Impact (`80/20` Stratified Split):**
  * `max_depth=1`: Train `0.67` | Test `0.61` (Underfitting)
  * `max_depth=2`: Train `0.96` | Test `0.92`
  * `max_depth=3`: Train `0.98` | Test `0.94` *(Optimal single-tree depth)*
  * `max_depth=None`: Train `1.00` | Test `0.89` *(Overfitting)*
* **Repeated Stratified Cross-Validation (`5-Fold × 10 Repeats`):**
  * **Majority Baseline:** `0.399 ± 0.043`
  * **Decision Tree (unpruned):** `0.895 ± 0.071`
  * **Decision Tree (`max_depth=3`):** `0.916 ± 0.058`
  * **Bagging (`m = p`):** `0.973 ± 0.038`
  * **Random Forest (`max_features="sqrt"`):** **`0.983 ± 0.029`** *(Best overall accuracy & lowest variance)*
* **Noise-Column Sanity Check:** Adding a synthetic `random_noise` feature showed slightly inflated impurity-based importance (`0.0121`), whereas **permutation importance** dropped to `-0.0053`, accurately confirming zero true predictive value.

## 🛠️ Tech Stack

* **Language:** Python 3.x
* **Libraries:** `scikit-learn`, `numpy`, `pandas`, `matplotlib`

## 🚀 Getting Started

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/mdanwarulkarim/Decision_Tree_Random_Forest_Wine_dataset_ML.git](https://github.com/mdanwarulkarim/Decision_Tree_Random_Forest_Wine_dataset_ML.git)
   cd Decision_Tree_Random_Forest_Wine_dataset_ML
