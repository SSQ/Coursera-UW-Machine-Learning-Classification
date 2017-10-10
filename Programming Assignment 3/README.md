# Goal
Implement your own logistic regression classifier with L2 regularization.
# File Description
- `.zip` files is data file.
  - [`amazon_baby_subset.zip`](https://github.com/SSQ/Coursera-UW-Machine-Learning-Classification/blob/master/Programming%20Assignment%202/amazon_baby_subset.zip) (unzip `amazon_baby_subset.csv`) 53072 sentiments (26579 positive, 26493 negative).
- `.ipynb` file is the solution of Week 2 program assignment 3
  - `Logistic Regression with L2 regularization.ipynb`
- `.html` file is the html version of `.ipynb` file.
  - `Logistic+Regression+with+L2+regularization.html`
# Snapshot
- **Recommend** open `md` file inside a file
- open `.html` file via brower for quick look.
# Algorithm
- Logistic Regression with L2 regularization
# Implementation
- Extract features from Amazon product reviews.
- Convert an dataframe into a NumPy array.
- Write a function to compute the derivative of log likelihood function with an L2 penalty with respect to a single coefficient.
- Implement gradient ascent with an L2 penalty.
- Empirically explore how the L2 penalty can ameliorate overfitting.
# Implementation in detail
- Load and process review dataset
  - Load the dataset into a data frame named products.
  - data transformations
  - Compute word counts (only for [important_words](Compute word counts (only for important_words)))
- Train-Validation split
  - split the data into a train-validation split with 80%, here we use provided [training](https://github.com/SSQ/Coursera-UW-Machine-Learning-Classification/blob/master/Programming%20Assignment%203/module-4-assignment-train-idx.json) and [validation](https://github.com/SSQ/Coursera-UW-Machine-Learning-Classification/blob/master/Programming%20Assignment%203/module-4-assignment-validation-idx.json) index
  - Convert train_data and validation_data into multi-dimensional arrays.
- Building on logistic regression with no L2 penalty assignment
  - Compute predictions given by the link function
- Adding L2 penalty
  - Adding L2 penalty to the derivative
  - computing log likelihood with L2
  - Write a function `logistic_regression_with_L2` to fit a logistic regression model under L2 regularization.
- Explore effects of L2 regularization
  - train models with different L2 (0, 4, 10, 1e2, 1e3, 1e5)
- Compare coefficients
  - Analysis coefficient without penalty
  - observe the effect of increasing L2 penalty on the 10 words
- Measuring accuracy
  - compute the accuracy of the classifier model. (**training 0.79, validation 0.78**)
  
