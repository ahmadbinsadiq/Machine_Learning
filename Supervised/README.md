# Supervised Learning Algorithms Repository

## Overview
This repository aims to include a comprehensive collection of supervised learning algorithms and their subcategories. The goal is to provide a wide range of implementations to demonstrate various techniques in regression and classification. Each algorithm is implemented in Python and comes with detailed documentation and example usage.

## Algorithms Included
1. **Regression**
   - Simple Linear Regression
   - Multiple Linear Regression
   - Ridge Regression
   - Lasso Regression
   - Polynomial Regression

2. **Classification**
   - Logistic Regression
   - Decision Trees
   - Random Forest
   - Support Vector Machines (SVM)
   - K-Nearest Neighbors (KNN)
   - Naive Bayes

## Short Summary
| Technique            | Name                     | Import & Implementation Code                                                                                                                                                                    | Implementation Details and Options                                                                                                                                                                                                                                                                                                                                                               |
|----------------------|--------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Regression           | Simple Linear Regression  | `from sklearn.linear_model import LinearRegression` <br> `regressor = LinearRegression()`                                                                                                          | -                                                                                                                                                                                                                                                                                                                                                                                                |
|                      | Multiple Linear Regression| `from sklearn.linear_model import LinearRegression` <br> `regressor = LinearRegression()`                                                                                                          | -                                                                                                                                                                                                                                                                                                                                                                                                |
|                      | Ridge Regression         | `from sklearn.linear_model import Ridge` <br> `ridge_regressor = Ridge(alpha=1.0)`                                                                                                                | **Options:**<br> `alpha`: Regularization strength. Higher values increase the regularization effect.<br> `solver`: Algorithm to use in the optimization problem ('auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga').                                                                                                                                                                                                                             |
|                      | Lasso Regression         | `from sklearn.linear_model import Lasso` <br> `lasso = Lasso(alpha=0.1)`                                                                                                                         | **Options:**<br> `alpha`: Regularization parameter. Higher values increase regularization strength and shrink coefficients.<br> `max_iter`: Maximum number of iterations for optimization.<br> `selection`: Method used to select features ('cyclic', 'random').                                                                                                                                                                                         |
|                      | Polynomial Regression    | `from sklearn.preprocessing import PolynomialFeatures` <br> `poly = PolynomialFeatures(degree=2)` <br> `x= poly.fit_transform(X_train)` <br> `y=poly.transform(X_test)` <br> `from sklearn.linear_model import LinearRegression` <br> `regressor = LinearRegression()` | **Options:**<br> `degree`: Degree of polynomial features. Higher degrees capture more complex relationships.<br> `interaction_only`: Whether to include only interaction features (True) or all polynomial features (False).<br> `include_bias`: Whether to include a bias column (True) in polynomial features.                                                                                                                                                      |
| Classification       | Logistic Regression      | `from sklearn.linear_model import LogisticRegression` <br> `log_reg = LogisticRegression()`                                                                                                      | **Options:**<br> `penalty`: Norm used in penalization ('l1', 'l2', 'elasticnet', 'none').<br> `C`: Inverse of regularization strength.<br> `solver`: Algorithm to use ('newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga').                                                                                                                                                                                                                                        |
|                      | Decision Trees           | `from sklearn.tree import DecisionTreeClassifier` <br> `tree_classifier = DecisionTreeClassifier(random_state=42)`                                                                              | **Options:**<br> `criterion`: Measure of split quality ('gini', 'entropy').<br> `max_depth`: Maximum depth of the tree.<br> `min_samples_split`: Minimum samples required to split an internal node.<br> `min_samples_leaf`: Minimum samples required at a leaf node.<br> `max_features`: Number of features to consider when looking for the best split.                                                                                                                                                       |
|                      | Random Forest            | `from sklearn.ensemble import RandomForestClassifier` <br> `rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)`                                                         | **Options:**<br> `n_estimators`: Number of trees in the forest.<br> `criterion`: Measure of split quality ('gini', 'entropy').<br> `max_depth`: Maximum depth of the trees.<br> `min_samples_split`: Minimum samples required to split an internal node.<br> `min_samples_leaf`: Minimum samples required at a leaf node.<br> `max_features`: Number of features to consider when looking for the best split.                                                                                           |
|                      | Support Vector Machines (SVM) | `from sklearn.svm import SVC` <br> `svc_classifier = SVC(kernel='linear', random_state=42)`                                                                                                     | **Options:**<br> `kernel`: Specifies the kernel type ('linear', 'poly', 'rbf', 'sigmoid').<br> `C`: Regularization parameter.<br> `gamma`: Kernel coefficient for 'rbf', 'poly', and 'sigmoid'.                                                                                                                                                                                                                                                                      |
|                      | K-Nearest Neighbors (KNN)| `from sklearn.neighbors import KNeighborsClassifier` <br> `knn_classifier = KNeighborsClassifier(n_neighbors=5)`                                                                              | **Options:**<br> `n_neighbors`: Number of neighbors to use for classification.<br> `weights`: Weight function ('uniform', 'distance').<br> `algorithm`: Algorithm used to compute neighbors ('auto', 'ball_tree', 'kd_tree', 'brute').                                                                                                                                                                                                                                 |
|                      | Naive Bayes              | `from sklearn.naive_bayes import GaussianNB` <br> `nb_classifier = GaussianNB()`                                                                                                                | -                                                                                                                                                                                                                                                                                                                                                                                                |


## Dataset
The algorithms are demonstrated using various datasets sourced from reliable sources like Kaggle, UCI Machine Learning Repository, and other publicly available datasets. Each dataset is preprocessed as necessary for the corresponding algorithm.

## Repository Structure
- **data/**: Contains the datasets used in the examples.
- **regression/**: Implementation scripts for regression algorithms.
  - *linear_regression.ipynb*: Implementation of linear regression with example usage.
  - *ridge_regression.ipynb*: Implementation of ridge regression with example usage.
  - *lasso_regression.ipynb*: Implementation of lasso regression with example usage.
  - *polynomial_regression.ipynb*: Implementation of polynomial regression with example usage.
- **classification/**: Implementation scripts for classification algorithms.
  - *logistic_regression.ipynb*: Implementation of logistic regression with example usage.
  - *decision_trees.ipynb*: Implementation of decision trees with example usage.
  - *random_forest.ipynb*: Implementation of random forest with example usage.
  - *svm.ipynb*: Implementation of support vector machines with example usage.
  - *knn.ipynb*: Implementation of k-nearest neighbors with example usage.
  - *naive_bayes.ipynb*: Implementation of naive bayes with example usage.
- **utils/**: Utility scripts for data preprocessing, evaluation metrics, etc.
- **README.md**: Detailed information about the repository, installation instructions, usage examples, and more.

## Usage
To use any of the algorithms, follow these steps:
1. Clone the repository:
   ```bash
   git clone https://github.com/ahmadbinsadiq/Machine_Learning.git
