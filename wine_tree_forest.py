"""
Decision Tree & Random Forest on the Wine dataset  -  companion script
Author: Md. Anwarul Karim | MS in Data Science
Run:  pip install scikit-learn pandas matplotlib   then   python wine_tree_forest.py
Every step matches pages 14-19 of the PDF (scikit-learn 1.8.0, random_state = 42).
"""
import warnings
warnings.filterwarnings("ignore")


# ======================================================================
# STEP 1-2 | load the data and split it
# ======================================================================
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

wine = load_wine(as_frame=True)
X, y = wine.data, wine.target
print(X.shape, "missing:", X.isna().sum().sum())
print("class counts:", np.bincount(y))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)
print(X_train.shape, X_test.shape, np.bincount(y_train))


# ======================================================================
# STEP 3 | grow a tree and watch it overfit
# ======================================================================
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
print("Train:", dt.score(X_train, y_train),
      "| Test:", round(dt.score(X_test, y_test), 4))
print("Depth:", dt.get_depth(), "| Leaves:", dt.get_n_leaves())

for d in [1, 2, 3, 4, 6, None]:
    m = DecisionTreeClassifier(max_depth=d, random_state=42)
    m.fit(X_train, y_train)
    print(d, round(m.score(X_train, y_train), 3), round(m.score(X_test, y_test), 3))


# ======================================================================
# STEP 4 | read the tree (plot_tree)
# ======================================================================
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

dt3 = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_train, y_train)

plt.figure(figsize=(15, 7))
plot_tree(dt3, feature_names=X.columns, class_names=wine.target_names,
          filled=True, rounded=True)
plt.show()


# ======================================================================
# STEP 5 | evaluate: tree vs forest
# ======================================================================
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

rf = RandomForestClassifier(n_estimators=100, oob_score=True,
                            random_state=42, n_jobs=-1).fit(X_train, y_train)
for name, model in [("Tree", dt3), ("Forest", rf)]:
    pred = model.predict(X_test)
    print(name, "accuracy:", round(accuracy_score(y_test, pred), 4))
    print(confusion_matrix(y_test, pred).tolist())
print("Forest OOB score:", round(rf.oob_score_, 4))

votes = [int(t.predict(X_test.iloc[[0]].values)[0]) for t in rf.estimators_]
print("votes (c0, c1, c2):", np.bincount(votes, minlength=3))


# ======================================================================
# STEP 6 | compare models fairly (repeated CV) + n_estimators
# ======================================================================
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.dummy import DummyClassifier

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
models = {
    "Majority baseline": DummyClassifier(),
    "Tree (unpruned)": DecisionTreeClassifier(random_state=42),
    "Tree (depth 3)": DecisionTreeClassifier(max_depth=3, random_state=42),
    "Bagging (m = p)": RandomForestClassifier(max_features=None, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Forest, no randomness": RandomForestClassifier(
        max_features=None, bootstrap=False, random_state=42),
}
for name, m in models.items():
    s = cross_val_score(m, X, y, cv=cv, n_jobs=-1)
    print(f"{name:22s} {s.mean():.3f} ± {s.std():.3f}")

for k in [10, 25, 50, 100, 200, 500]:
    m = RandomForestClassifier(n_estimators=k, oob_score=True, random_state=42, n_jobs=-1)
    print(k, round(m.fit(X_train, y_train).oob_score_, 3), end=" | ")
print()


# ======================================================================
# STEP 7 | feature importance and the noise-column test
# ======================================================================
from sklearn.inspection import permutation_importance
Xn = X.assign(random_noise=np.random.RandomState(42).rand(len(X)))
Xn_tr, Xn_te, yn_tr, yn_te = train_test_split(
    Xn, y, test_size=0.20, random_state=42, stratify=y)
rf2 = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1).fit(Xn_tr, yn_tr)
imp = pd.Series(rf2.feature_importances_, index=Xn.columns)
pi = permutation_importance(rf2, Xn_te, yn_te, n_repeats=30, random_state=42, n_jobs=-1)
perm = pd.Series(pi.importances_mean, index=Xn.columns)
print("noise column -> impurity:", round(imp["random_noise"], 4),
      "| permutation:", round(perm["random_noise"], 4))
