# Unsupervised Learning Algorithms Repository

## Overview
This repository aims to include a comprehensive collection of unsupervised learning algorithms and their subcategories. The goal is to provide a wide range of implementations to demonstrate various techniques in clustering, dimensionality reduction, and anomaly detection. Each algorithm is implemented in Python and comes with detailed documentation and example usage.

## Algorithms Included

### Clustering

- K-Means Clustering
- Hierarchical Clustering
- DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
- Gaussian Mixture Models

### Dimensionality Reduction

- Principal Component Analysis (PCA)
- Independent Component Analysis (ICA)
- t-Distributed Stochastic Neighbor Embedding (t-SNE)
- Autoencoders

### Anomaly Detection

- Isolation Forest
- Local Outlier Factor (LOF)
- One-Class SVM

## Short Description
| Technique                    | Name                                        | Import & Implementation Code                                                                                                  | Implementation Details and Options                                                                                                                                                                                                                                                                                                                                                                     |
|------------------------------|---------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Clustering                   | K-Means Clustering                           | `from sklearn.cluster import KMeans` <br> `kmeans = KMeans(n_clusters=2, random_state=42)`                                      | **Options:**<br> `n_clusters`: Number of clusters to form.<br> `init`: Method for initialization ('k-means++', 'random').<br> `max_iter`: Maximum number of iterations.<br> `algorithm`: Algorithm to use ('auto', 'full', 'elkan').                                                                                                                                                                                                                                    |
|                              | Hierarchical Clustering                      | `from sklearn.cluster import AgglomerativeClustering` <br> `agg_clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=10, linkage='ward')`                                | **Options:**<br> `n_clusters`: Number of clusters to form if not None.<br> `distance_threshold`: The linkage distance threshold above which clusters will not be merged.<br> `linkage`: Linkage criterion ('ward', 'complete', 'average', 'single').                                                                                                                                                                                                                     |
|                              | DBSCAN (Density-Based Spatial Clustering)    | `from sklearn.cluster import DBSCAN` <br> `dbscan = DBSCAN(eps=0.5, min_samples=5)`                                               | **Options:**<br> `eps`: Maximum distance between two samples to be considered in the same neighborhood.<br> `min_samples`: Minimum number of samples in a neighborhood for a point to be considered as a core point.<br> `metric`: Distance metric to use ('euclidean', 'manhattan', 'cosine', etc.).                                                                                                                                                             |
|                              | Gaussian Mixture Models                     | `from sklearn.mixture import GaussianMixture` <br> `gmm = GaussianMixture(n_components=3, random_state=42)`                       | **Options:**<br> `n_components`: Number of Gaussian components in the mixture.<br> `covariance_type`: Covariance matrix type ('full', 'tied', 'diag', 'spherical').<br> `max_iter`: Maximum number of EM (Expectation-Maximization) iterations.<br> `init_params`: Method for initialization parameters ('kmeans', 'random').                                                                                                                                                      |
| Dimensionality Reduction     | Principal Component Analysis (PCA)           | `from sklearn.decomposition import PCA` <br> `pca = PCA(n_components=3)`                                                        | **Options:**<br> `n_components`: Number of components to keep.<br> `svd_solver`: Solver for decomposition ('auto', 'full', 'arpack', 'randomized').                                                                                                                                                                                                                                                                                                                    |
|                              | Independent Component Analysis (ICA)         | `from sklearn.decomposition import FastICA` <br> `ica = FastICA(n_components=3, random_state=42)`                                | **Options:**<br> `n_components`: Number of components to keep.<br> `algorithm`: Algorithm to use ('parallel', 'deflation').                                                                                                                                                                                                                                                                                                                                          |
|                              | t-Distributed Stochastic Neighbor Embedding (t-SNE) | `from sklearn.manifold import TSNE` <br> `tsne = TSNE(n_components=2, random_state=42)`                                         | **Options:**<br> `n_components`: Number of dimensions in the embedded space.<br> `perplexity`: Perplexity value affects the number of nearest neighbors used in other manifold learning algorithms.<br> `learning_rate`: Learning rate for the optimization (usually between 10 and 1000).<br> `n_iter`: Maximum number of iterations for the optimization.                                                                                                                                                        |
| Anomaly Detection            | Isolation Forest                            | `from sklearn.ensemble import IsolationForest` <br> `iso_forest = IsolationForest(contamination=0.1, random_state=42)`           | **Options:**<br> `n_estimators`: Number of base estimators (trees) in the ensemble.<br> `max_samples`: Number of samples to draw from X to train each base estimator.<br> `contamination`: Expected proportion of outliers in the data set (used when fitting to define the threshold on the decision function).                                                                                                                                                             |
|                              | Local Outlier Factor (LOF)                  | `from sklearn.neighbors import LocalOutlierFactor` <br> `lof = LocalOutlierFactor(n_neighbors=20, contamination=0.001)`            | **Options:**<br> `n_neighbors`: Number of neighbors to use for outlier detection.<br> `contamination`: Expected proportion of outliers in the data set.<br> `metric`: Distance metric to use ('euclidean', 'manhattan', 'minkowski').                                                                                                                                                                                                                                    |
|                              | One-Class SVM                               | `from sklearn.svm import OneClassSVM` <br> `ocsvm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.01)`                            | **Options:**<br> `kernel`: Specifies the kernel type ('linear', 'poly', 'rbf', 'sigmoid').<br> `gamma`: Kernel coefficient for 'rbf', 'poly', and 'sigmoid'.<br> `nu`: An upper bound on the fraction of training errors and a lower bound of the fraction of support vectors.                                                                                                                                                                                          |

## Dataset
The algorithms are demonstrated using various datasets sourced from reliable sources like Kaggle, UCI Machine Learning Repository, and other publicly available datasets. Each dataset is preprocessed as necessary for the corresponding algorithm.


- **clustering/**: Implementation scripts for clustering algorithms.
  - **kmeans_clustering.ipynb**: Implementation of K-Means Clustering with example usage.
  - **hierarchical_clustering.ipynb**: Implementation of Hierarchical Clustering with example usage.
  - **dbscan_clustering.ipynb**: Implementation of DBSCAN with example usage.
  - **gmm_clustering.ipynb**: Implementation of Gaussian Mixture Models with example usage.
- **dimensionality_reduction/**: Implementation scripts for dimensionality reduction algorithms.
  - **pca.ipynb**: Implementation of Principal Component Analysis with example usage.
  - **ica.ipynb**: Implementation of Independent Component Analysis with example usage.
  - **tsne.ipynb**: Implementation of t-SNE with example usage.
  - **autoencoders.ipynb**: Implementation of Autoencoders with example usage.
- **anomaly_detection/**: Implementation scripts for anomaly detection algorithms.
  - **isolation_forest.ipynb**: Implementation of Isolation Forest with example usage.
  - **lof.ipynb**: Implementation of Local Outlier Factor with example usage.
  - **one_class_svm.ipynb**: Implementation of One-Class SVM with example usage.
- **utils/**: Utility scripts for data preprocessing, evaluation metrics, and visualization tools.
- **README.md**: Detailed information about the repository, installation instructions, usage examples, and more.

## Usage
To use any of the algorithms, follow these steps:

1. Clone the repository:
   ```sh
   git clone https://github.com/ahmadbinsadiq/Machine_Learning.git
