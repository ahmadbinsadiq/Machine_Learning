# Semi-Supervised Learning Algorithms Repository

## Overview

This repository aims to provide a comprehensive collection of semi-supervised learning algorithms and their implementations. The goal is to demonstrate various techniques that leverage both labeled and unlabeled data to improve model performance. Each algorithm is implemented in Python and comes with detailed documentation and example usage.

## Algorithms Included

### Self-Training

Self-training is a semi-supervised learning technique where a model is initially trained on labeled data. It then uses this model to predict labels for unlabeled data, which are added to the labeled dataset for subsequent training iterations. This process iterates until convergence or a predefined stopping criterion.

### Co-Training

Co-training involves training multiple classifiers on different subsets of features or views of the data. Initially, each classifier is trained on labeled data. They then collaborate by exchanging confidently predicted labels for unlabeled data instances between each other. This exchange process continues iteratively, enhancing the overall learning process.

### Tri-Training

Tri-training extends co-training by using three base classifiers instead of two. Each classifier initially trains on the labeled data and then exchanges predictions on unlabeled data instances. The agreement among the classifiers on these predictions determines whether the instances are added to the labeled dataset for subsequent iterations.

## Contents

- **self_training/**: Contains code and examples for Self-Training.
  - **self_training.ipynb**: Implementation of Self-Training with example usage.
  
- **co_training/**: Contains code and examples for Co-Training.
  - **co_training.ipynb**: Implementation of Co-Training with example usage.
  
- **tri_training/**: Contains code and examples for Tri-Training.
  - **tri_training.ipynb**: Implementation of Tri-Training with example usage.

## Getting Started

To get started with any of the semi-supervised learning techniques, navigate to the respective folder and follow the instructions provided in the individual README files. Each folder contains example code, data, and detailed explanations to help you understand and apply the specific semi-supervised learning techniques.

## Contributing

If you have any improvements or additional examples, feel free to create a pull request. Contributions are welcome!

## Author

* **LinkedIn:** [Ahmad Bin Sadiq](https://www.linkedin.com/in/ahmad-bin-sadiq/)
* **Email:** ahmadbinsadiq@gmail.com

Happy learning and coding!
