# ML-Crash_Course-Cheat_Sheet-with-Code-Implemention

A hands-on reference notebook covering core Machine Learning algorithms, hyperparameter tuning, model evaluation, and end-to-end scikit-learn pipelines.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Topics & Algorithms Covered](#topics--algorithms-covered)
  - [Supervised Learning: Regression](#1-supervised-learning-regression)
  - [Supervised Learning: Classification](#2-supervised-learning-classification)
  - [Unsupervised Learning](#3-unsupervised-learning)
  - [Dimensionality Reduction](#4-dimensionality-reduction)
  - [End-to-End ML Pipeline](#5-end-to-end-ml-pipeline)
  - [Planned / Upcoming Sections](#6-planned--upcoming-sections)
- [Requirements & Installation](#requirements--installation)
- [How to Use](#how-to-use)

---

## 📖 Overview

This notebook serves as a practical cheatsheet and quick-start guide for machine learning practitioners. Each section contains:
- Clear theoretical explanations of the algorithm or step.
- Self-contained code implementations using popular libraries (`scikit-learn`, `xgboost`, `lightgbm`, `catboost`).
- Best practices for data preprocessing, scaling, hyperparameter tuning (`GridSearchCV`), and metric evaluation.

---

## 🧠 Topics & Algorithms Covered

### 1. Supervised Learning: Regression
* **Simple & Multiple Linear Regression:** Basic line fitting and multi-feature modeling.
* **Regularized Linear Models:** Ridge ($L_2$), Lasso ($L_1$), and ElasticNet with `GridSearchCV` hyperparameter optimization.
* **Polynomial Regression:** Nonlinear modeling using `PolynomialFeatures` inside a scikit-learn pipeline.
* **Decision Tree Regressor:** Tree-based regression with depth and split parameter tuning.
* **Support Vector Regression (SVR):** Linear, RBF, and polynomial kernel regression with $C$ and $\epsilon$ tuning.
* **Random Forest Regressor:** Ensemble bagging regression and Out-Of-Bag (OOB) scoring.

### 2. Supervised Learning: Classification
* **Logistic Regression:** Regularized classification with feature scaling (`StandardScaler`).
* **K-Nearest Neighbors (KNN):** Distance-based classification and neighborhood search tuning.
* **Naive Bayes:** 
  * `GaussianNB` for continuous numerical features.
  * `MultinomialNB` paired with `CountVectorizer` for NLP text classification.
* **Decision Tree Classifier:** Gini impurity / entropy-based splits with pruning.
* **Ensemble & Boosting:**
  * **Random Forest Classifier:** Bagging ensemble with hyperparameter exploration.
  * **Gradient Boosting:** Sequential residual correction via `GradientBoostingClassifier`.
  * **AdaBoost:** Adaptive boosting with decision stump estimators.
  * **XGBoost:** Scalable gradient boosted decision trees (`XGBClassifier`).
  * **LightGBM:** Fast, histogram-based gradient boosting (`LGBMClassifier`).
  * **CatBoost:** High-performance gradient boosting supporting categorical features (`CatBoostClassifier`).

### 3. Unsupervised Learning
* **K-Means Clustering:** Partition-based clustering and centroid visualization.
* **Agglomerative Hierarchical Clustering:** Bottom-up clustering with dendrogram generation using `scipy`.
* **DBSCAN:** Density-based clustering capable of discovering arbitrary cluster shapes and handling noise.
* **Gaussian Mixture Models (GMM):** Probabilistic clustering with soft assignment.

### 4. Dimensionality Reduction
* **Principal Component Analysis (PCA):** Unsupervised variance maximization and feature projection.
* **Linear Discriminant Analysis (LDA):** Supervised class-separability projection combined with classification pipelines.

### 5. End-to-End ML Pipeline
* Complete automated workflow demonstrating:
  * Mixed-type data handling with `ColumnTransformer` (numerical scaling + categorical one-hot encoding).
  * Pipeline chaining with model estimators (`RandomForestClassifier`).
  * Serialization and persistence (`joblib.dump` and `joblib.load`) for deployment and inference.

### 6. Planned / Upcoming Sections
* **Forecasting Models:** ARIMA, SARIMA, and Holt-Winters Exponential Smoothing.
* **Comprehensive Evaluation & Validation:** Confusion matrices, ROC-AUC, bias-variance tradeoffs, and cross-validation strategies.

---

## 🛠 Requirements & Installation

To run the notebook locally or in an interactive environment, install the required packages:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm catboost scipy
🚀 How to Use
Open ML_Crash Course & CheatSheet with Code.ipynb in Google Colab or JupyterLab.

Run cells sequentially from top to bottom.

Modify hyperparameter grids or swap in your own datasets to test different model architectures.
