# Goal
The goal of this assignment is to implement your own logistic regression classifier
# File Description
- `.zip` files is data file.
  - `amazon_baby_subset.zip` (unzip `amazon_baby_subset.csv`) consists of 53072 sentiments (26579 positive, 26493 negative)
- `.ipynb` file is the solution of Week 2 program assignment 2
  - `Implementing logistic regression from scratch.ipynb`
- `.html` file is the html version of `.ipynb` file.
  - `Implementing+logistic+regression+from+scratch.html`
# Snapshot
open `.html` file via brower for quick look.
# Algorithm
- Logistic Regression
# Implementation
- Extract features from Amazon product reviews.
- Convert an SFrame into a NumPy array.
- Implement the link function for logistic regression.
- Write a function to compute the derivative of the log likelihood function with respect to a single coefficient.
- Implement gradient ascent.
- Given a set of coefficients, predict sentiments.
- Compute classification accuracy for the logistic regression model.

# Implementation in detail
- Load data
- Perform text cleaning. removing punctuation, fill in N/A's in the empty review (pandas.fillna({'review':''}))
- load the [important features](https://github.com/SSQ/Coursera-UW-Machine-Learning-Classification/blob/master/Programming%20Assignment%202/important_words.json)
- compute a count for the number of times the important word occurs in the review 
- convert our data frame to a multi-dimensional array with `Pandas as_matrix()` function
- Compute predictions given by the link function
- computes the derivative of log likelihood with respect to a single coefficient w_j
- Write a function `compute_log_likelihood`
- Write a function `logistic_regression` to fit a logistic regression model using gradient ascent
- compute class predictions
- Measuring accuracy (0.75)
- Compute the 10 most positive words and negative words
-
