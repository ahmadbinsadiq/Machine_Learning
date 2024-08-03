# Regression in Machine Learning

This folder contains various types of regression analyses used in machine learning. Regression analysis is a fundamental technique used to understand the relationship between a dependent (response) variable and one or more independent (predictor) variables. Below is a table outlining when to use each type of regression.

## Types of Regression and Their Use Cases

| Regression Type             | When to Use                                                                                          |
|-----------------------------|------------------------------------------------------------------------------------------------------|
| **Simple Linear Regression**| When you have one predictor variable and want to model the linear relationship between the predictor and the response variable. |
| **Multiple Linear Regression** | When you have multiple predictor variables and want to model the linear relationship between these predictors and the response variable. |
| **Ridge Regression**        | When you have multiple predictor variables and multicollinearity (high correlation between predictors) is present. Ridge regression adds a penalty to the size of the coefficients, which helps to prevent overfitting. |
| **Lasso Regression**        | When you want to perform feature selection by shrinking some coefficients to zero, thus removing some predictors from the model. Lasso is useful when you have many predictor variables, and you suspect that some are not contributing much to the prediction. |
| **Polynomial Regression**   | When the relationship between the predictor and the response variable is not linear, and you want to model this non-linear relationship by including polynomial terms (squared, cubed, etc.) of the predictor variables. |

## Contents

- `simple_linear_regression/`: Contains code and example for Simple Linear Regression.
- `multiple_linear_regression/`: Contains code and example for Multiple Linear Regression.
- `ridge_regression/`: Contains code and example for Ridge Regression.
- `lasso_regression/`: Contains code and example for Lasso Regression.
- `polynomial_regression/`: Contains code and example for Polynomial Regression.

## Getting Started

To get started with any of the regression types, navigate to the respective folder and follow the instructions provided in the individual README files. Each folder contains example code, data, and detailed explanations to help you understand and apply the specific regression techniques.

## Contributing

If you have any improvements or additional examples, feel free to create a pull request. Contributions are welcome!

## Author

* **LinkedIn:** [Ahmad Bin Sadiq](https://www.linkedin.com/in/ahmad-bin-sadiq/)
* **Email:** ahmadbinsadiq@gmail.com

---

Happy coding!
