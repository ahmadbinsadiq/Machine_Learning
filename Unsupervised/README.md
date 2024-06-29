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

## Dataset
The algorithms are demonstrated using various datasets sourced from reliable sources like Kaggle, UCI Machine Learning Repository, and other publicly available datasets. Each dataset is preprocessed as necessary for the corresponding algorithm.

## Repository Structure

unsupervised-learning

│
├── clustering/
│ ├── kmeans_clustering.ipynb
│ ├── hierarchical_clustering.ipynb
│ ├── dbscan_clustering.ipynb
│ └── gmm_clustering.ipynb
│
├── dimensionality_reduction/
│ ├── pca.ipynb
│ ├── ica.ipynb
│ ├── tsne.ipynb
│ └── autoencoders.ipynb
│
├── anomaly_detection/
│ ├── isolation_forest.ipynb
│ ├── lof.ipynb
│ └── one_class_svm.ipynb
│
├── utils/
│ ├── data_preprocessing.py
│ ├── evaluation_metrics.py
│ └── visualization_tools.py
│
└── README.md

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
   git clone https://github.com/ahmadbinsadiq/Machine_Learning/tree/e36d7eacc7580fdc061224f84b06517d94c77f6a/Unsupervised
