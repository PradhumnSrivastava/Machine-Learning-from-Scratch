---

title: Linear Regression

tags:

- MachineLearning
    
- SupervisedLearning
    
- Regression
    
- LinearRegression
    

aliases:

- Ordinary Least Squares
    
- OLS
    

created: 2026-06-24  
updated: 2026-06-24

status: Complete  
difficulty: Beginner  
category: Regression Algorithm

prerequisites:

- Calculus
    
- Linear Algebra
    
- Probability
    
- Mean Squared Error
    

related:

- Gradient Descent
    
- Ridge Regression
    
- Lasso Regression
    

---

# Linear Regression

> [!quote]  
> **"Linear Regression is one of the simplest and most fundamental Machine Learning algorithms used to model the relationship between dependent and independent variables."**

---

# 1. Introduction

Linear Regression is a **Supervised Machine Learning algorithm** used for predicting **continuous numerical values**.

Examples:

```text
House Price Prediction
Salary Prediction
Stock Price Prediction
Temperature Forecasting
Sales Forecasting
```

The objective of Linear Regression is:

> To find the best-fit line that explains the relationship between input features and the target variable.

---

# 2. Mathematical Intuition

Suppose:

```text
Hours Studied → Exam Marks
```

Generally:

```text
More Study Hours → Higher Marks
```

This relationship can be represented using a straight line.

## Equation of Straight Line

$$  
y = mx + c  
$$

Where:

|Symbol|Meaning|
|---|---|
|y|Dependent Variable (Target)|
|x|Independent Variable (Feature)|
|m|Slope of Line|
|c|Intercept|

---

## Machine Learning Form

$$  
\hat y = \beta_0+\beta_1x  
$$

Where:

|Symbol|Meaning|
|---|---|
|$\hat y$|Predicted Value|
|$\beta_0$|Intercept (Bias)|
|$\beta_1$|Coefficient (Weight)|
|x|Input Feature|

---

## Multiple Linear Regression

For multiple features:

$$  
\hat y=\beta_0+\beta_1x_1+\beta_2x_2+\cdots+\beta_nx_n  
$$

Matrix Form:

$$  
\hat Y=X\beta  
$$

Where:

|Symbol|Meaning|
|---|---|
|X|Feature Matrix|
|β|Weight Vector|
|Y|Target Vector|

---

# 3. Assumptions of Linear Regression

Linear Regression works well only when these assumptions are satisfied.

## 1. Linearity

Independent and dependent variables should have a linear relationship.

```text
x ↑ → y ↑
```

---

## 2. Independence of Errors

Residuals should be independent.

Mathematically:

$$  
Cov(\epsilon_i,\epsilon_j)=0  
$$

---

## 3. Homoscedasticity

Variance of residuals should remain constant.

Good:

```text
| | | | | |
```

Bad:

```text
/\
 \ \
  \  \
```

---

## 4. Normality of Residuals

Residuals should follow a normal distribution.

$$  
\epsilon\sim N(0,\sigma^2)  
$$

---

## 5. No Multicollinearity

Independent variables should not be highly correlated.

Example:

```text
Age and Years Lived
```

These two features are almost identical.

---

## 6. No Autocorrelation

Residuals should not be correlated over time.

Important for:

```text
Time Series Data
```

---

# 4. Working Principle

Linear Regression attempts to find the line:

```text
Best Fit Line
```

such that:

```text
Prediction Error is Minimum
```

---

## Goal

Find:

$$  
\beta_0,\beta_1,\beta_2,\cdots,\beta_n  
$$

such that:

$$  
Loss \rightarrow Minimum  
$$

---

Most commonly used loss:

$$  
MSE=\frac1N\sum_{i=1}^{N}(y_i-\hat y_i)^2  
$$

---

# 5. Mathematical Derivation

## Step 1: Hypothesis Function

For single feature:

$$  
\hat y_i=\beta_0+\beta_1x_i  
$$

---

## Step 2: Error

Residual:

$$  
e_i=y_i-\hat y_i  
$$

Substituting:

$$  
e_i=y_i-(\beta_0+\beta_1x_i)  
$$

---

## Step 3: Cost Function

We minimize:

# $$  
J(\beta_0,\beta_1)

\frac1N  
\sum_{i=1}^{N}  
(y_i-\hat y_i)^2  
$$

Substitute prediction:

# $$  
J(\beta_0,\beta_1)

\frac1N  
\sum_{i=1}^{N}  
(y_i-\beta_0-\beta_1x_i)^2  
$$

---

## Step 4: Ordinary Least Squares (OLS)

Take partial derivatives:

$$  
\frac{\partial J}{\partial \beta_0}=0  
$$

$$  
\frac{\partial J}{\partial \beta_1}=0  
$$

Solving:

### Slope

$$  
\beta_1=  
\frac{\sum(x_i-\bar x)(y_i-\bar y)}  
{\sum(x_i-\bar x)^2}  
$$

### Intercept

$$  
\beta_0=  
\bar y-\beta_1\bar x  
$$

---

# 6. Workflow (Prediction Process)

```text
Training Data
       ↓
Learn Relationship
       ↓
Estimate Coefficients
       ↓
Build Best Fit Line
       ↓
Receive New Data
       ↓
Apply Equation
       ↓
Generate Prediction
```

---

## Example

Training Data:

|Area|Price|
|---|---|
|1000|50|
|1200|60|
|1500|75|

Model learns:

$$  
Price=5+0.05(Area)  
$$

New house:

```text
Area = 1400
```

Prediction:

$$  
Price=5+0.05(1400)  
$$

$$  
Price=75  
$$

---

# 7. Parameters

Parameters are learned from data.

## 1. Intercept

$$  
\beta_0  
$$

Represents baseline prediction.

---

## 2. Coefficients

$$  
\beta_1,\beta_2,\cdots,\beta_n  
$$

Represent influence of each feature.

Example:

```text
Price = 10 + 5 × Area
```

Coefficient:

```text
5
```

means:

```text
1 unit increase in Area
increases Price by 5 units.
```

---

# 8. Hyperparameter Tuning in Linear Regression

Basic Linear Regression has very few hyperparameters.

---

## Important Hyperparameters

### 1. fit_intercept

```python
fit_intercept=True
```

Determines whether intercept should be calculated.

---

### 2. positive

```python
positive=True
```

Forces coefficients to be positive.

---

### 3. n_jobs

Controls parallel computation.

---

## Hyperparameter Tuning Example

```python
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression

params = {
    'fit_intercept':[True,False],
    'positive':[True,False]
}

grid = GridSearchCV(
    LinearRegression(),
    params,
    cv=5
)

grid.fit(X_train,y_train)

print(grid.best_params_)
```

---

# 9. Advantages

## 1. Simple and Easy to Understand

---

## 2. Computationally Efficient

---

## 3. Highly Interpretable

---

## 4. Fast Training

---

## 5. Works Well for Linear Relationships

---

## 6. Foundation of Many Advanced Algorithms

---

# 10. Disadvantages

## 1. Assumes Linearity

Cannot capture complex relationships.

---

## 2. Sensitive to Outliers

Extreme values distort the line.

---

## 3. Suffers from Multicollinearity

---

## 4. Underfits Complex Data

---

## 5. Requires Assumption Validation

---

# 11. Code Implementation

## Scikit-Learn Implementation

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Features and Target
X = df[['Area']]
y = df['Price']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model Creation
model = LinearRegression()

# Training
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("Coefficient:", model.coef_)
print("Intercept:", model.intercept_)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
```

---

## From Scratch Using NumPy

```python
import numpy as np

x = np.array([1,2,3,4,5])
y = np.array([2,4,5,4,5])

x_mean = np.mean(x)
y_mean = np.mean(y)

numerator = np.sum((x-x_mean)*(y-y_mean))
denominator = np.sum((x-x_mean)**2)

b1 = numerator/denominator
b0 = y_mean - b1*x_mean

print("Intercept:", b0)
print("Slope:", b1)
```

---

# 12. Evaluation Metrics

Common metrics:

```text
MAE
MSE
RMSE
R² Score
Adjusted R²
```

---

# 13. Real-World Applications

```text
House Price Prediction
Sales Forecasting
Demand Forecasting
Marketing Analytics
Economic Forecasting
Risk Analysis
```

---

# 14. Interview Questions

## Basic

1. What is Linear Regression?
    
2. What are the assumptions of Linear Regression?
    
3. What is the best-fit line?
    

---

## Intermediate

1. Explain OLS.
    
2. Difference between parameters and hyperparameters.
    
3. Why is MSE used?
    

---

## Advanced

1. Derive the OLS equation.
    
2. Explain multicollinearity.
    
3. Why does Linear Regression fail on non-linear data?
    

---

# 15. Summary

> [!summary]
> 
> - Linear Regression predicts continuous values.
>     
> - It models linear relationships between variables.
>     
> - It minimizes Mean Squared Error.
>     
> - Parameters are estimated using OLS or Gradient Descent.
>     
> - Assumptions must hold for reliable predictions.
>     
> - It is simple, interpretable, and widely used.
>     

---

# Mind Map

```text
Linear Regression
│
├── Supervised Learning
├── Continuous Prediction
├── Best Fit Line
├── OLS
├── MSE Minimization
├── Parameters
├── Assumptions
├── Evaluation Metrics
└── Prediction
```