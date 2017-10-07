# Goal
Implement a logistic regression classifier using stochastic gradient ascent
# File Description
- `.zip` files is data file.
  - [`amazon_baby_subset.zip`](https://github.com/SSQ/Coursera-UW-Machine-Learning-Classification/blob/master/Programming%20Assignment%202/amazon_baby_subset.zip) (unzip `amazon_baby_subset.csv`) 53072 sentiments (26579 positive, 26493 negative).
- `.ipynb` file is the solution of Week 7 program assignment 1
  - `Training Logistic Regression via Stochastic Gradient.ipynb`
- `.html` file is the html version of `.ipynb` file.
  - `Training+Logistic+Regression+via+Stochastic+Gradient.html`
- file
  - `Training+Logistic+Regression+via+Stochastic+Gradient+Ascent`
# Snapshot
- computer view. open `.html` file via brower for quick look.
- brower view. Training+Logistic+Regression+via+Stochastic+Gradient+Ascent / [Training Logistic Regression via Stochastic Gradient Ascent.md	](https://github.com/SSQ/Coursera-UW-Machine-Learning-Classification/blob/master/Week%207%20PA%201/Training%2BLogistic%2BRegression%2Bvia%2BStochastic%2BGradient%2BAscent/Training%20Logistic%20Regression%20via%20Stochastic%20Gradient%20Ascent.md)
# Algorithm
- stochastic gradient ascent
# Implementation
- Extract features from Amazon product reviews.
- Convert data into a NumPy array.
- Write a function to compute the derivative of log likelihood function (with L2 penalty) with respect to a single coefficient.
- Implement stochastic gradient ascent with L2 penalty
- Compare convergence of stochastic gradient ascent with that of batch gradient ascent
# Implementation in detail
- Load and process review dataset
  - Load the dataset into a data frame named products.
  - data transformations
  - Compute word counts (only for [important_words](Compute word counts (only for important_words)))
- Train-Validation split
  - split the data into a train-validation split with 80%, here we use provided [training](https://github.com/SSQ/Coursera-UW-Machine-Learning-Classification/blob/master/Programming%20Assignment%203/module-4-assignment-train-idx.json) and [validation](https://github.com/SSQ/Coursera-UW-Machine-Learning-Classification/blob/master/Programming%20Assignment%203/module-4-assignment-validation-idx.json) index
  - Convert train_data and validation_data into multi-dimensional arrays.
  ```python
    def get_numpy_data(dataframe, features, label):
        ...
        return(feature_matrix, label_array)
    ```
- Building on logistic regression  
```python
'''
feature_matrix: N * D(intercept term included)
coefficients: D * 1
predictions: N * 1
produces probablistic estimate for P(y_i = +1 | x_i, w).
estimate ranges between 0 and 1.
'''

def predict_probability(feature_matrix, coefficients):
    ...
    return predictions
```
- Derivative of log likelihood with respect to a single coefficient
```python
"""
errors: N * 1
feature: N * 1
derivative: 1 
"""
def feature_derivative(errors, feature):     
    ...
    return derivative
```
```python
def compute_avg_log_likelihood(feature_matrix, sentiment, coefficients):
    ...    
    return lp
```    
- Modifying the derivative for stochastic gradient ascent
- Modifying the derivative for using a batch of data points
- Averaging the gradient across a batch
- Implementing stochastic gradient ascent
```python
def logistic_regression_SG(feature_matrix, sentiment, initial_coefficients, step_size, batch_size, max_iter):
    ...
    return coefficients, log_likelihood_all
```
- Compare convergence behavior of stochastic gradient ascent
```python
coefficients, log_likelihood = logistic_regression_SG(feature_matrix_train, sentiment_train,\
                                        initial_coefficients=np.zeros(194),\
                                        step_size=5e-1, batch_size=1, max_iter=10)
```
```python
# YOUR CODE HERE
coefficients_batch, log_likelihood_batch = logistic_regression_SG(feature_matrix_train, sentiment_train,\
                                        initial_coefficients=np.zeros(194),\
                                        step_size=5e-1, batch_size=len(feature_matrix_train), max_iter=200)
```
- Make "passes" over the dataset
- Stochastic gradient ascent vs batch gradient ascent
- Explore the effects of step sizes on stochastic gradient ascent
- Plotting the log likelihood as a function of passes for each step size
