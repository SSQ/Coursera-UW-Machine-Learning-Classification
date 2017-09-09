# Coursera-UW-Machine-Learning-Classification

Course can be found [here](https://www.coursera.org/learn/ml-classification)

Notebook for quick search can be found [here](https://ssq.github.io/2017/08/19/Coursera%20UW%20Machine%20Learning%20Specialization%20Notebook/)

- Week 1: 
  - Linear Classifiers & Logistic Regression
    - decision boundaries
    - linear classifiers
    - class probability
    - logistic regression
    - impact of coefficient values on logistic regression output
    - 1-hot encoding
    - multiclass classification using the 1-versus-all
    - [x] [Predicting sentiment from product reviews](https://github.com/SSQ/Coursera-UW-Machine-Learning-Classification/tree/master/Programming%20Assignment%201)
- Week 2: 
  - Learning Linear Classifiers
    - Maximum likelihood estimation
    - Gradient ascent algorithm for learning logistic regression classifier
    - Choosing step size for gradient ascent/descent
    - (VERY OPTIONAL LESSON) Deriving gradient of logistic regression
    - [x] [Implementing logistic regression from scratch](https://github.com/SSQ/Coursera-UW-Machine-Learning-Classification/tree/master/Programming%20Assignment%202)  
  - Overfitting & Regularization in Logistic Regression
    - Overfitting in classification
    - Overconfident predictions due to overfitting
    - L2 regularized logistic regression
    - Sparse logistic regression
    - [x] [Implementing Logistic Regression with L2 regularization](https://github.com/SSQ/Coursera-UW-Machine-Learning-Classification/tree/master/Programming%20Assignment%203)
- Week 3:
  - Decision Trees
    - Predicting loan defaults with decision trees
    - Learning decision trees
        - Recursive greedy algorithm
        - Learning a decision stump
        - Selecting best feature to split on
        - When to stop recursing
    - Using the learned decision tree
        - Traverse a decision tree to make predictions: Majority class predictions; Probability predictions; Multiclass classification
    - Learning decision trees with continuous inputs
        - Threshold splits for continuous inputs
        - (OPTIONAL) Picking the best threshold to split on
    - [x] [Identifying safe loans with decision trees](https://github.com/SSQ/Coursera-UW-Machine-Learning-Classification/tree/master/Programming%20Assignment%204)
    - [x] [Implementing binary decision trees from scratch](https://github.com/SSQ/Coursera-UW-Machine-Learning-Classification/tree/master/Programming%20Assignment%205)
- Week 4
  - Overfitting in decision trees 
    - Identify when overfitting in decision trees
    - Prevent overfitting with early stopping
      - Limit tree depth
      - Do not consider splits that do not reduce classification error
      - Do not split intermediate nodes with only few points
    - Prevent overfitting by pruning complex trees
      - Use a total cost formula that balances classification error and tree complexity
      - Use total cost to merge potentially complex trees into simpler ones 
    - [x] [Decision Trees in Practice for preventing overfitting](https://github.com/SSQ/Coursera-UW-Machine-Learning-Classification/tree/master/Programming%20Assignment%206) 
  - Handling missing data 
    - Describe common ways to handling missing data:
      1. Skip all rows with any missing values
      2. Skip features with many missing values
      3. Impute missing values using other data points
    - Modify learning algorithm (decision trees) to handle missing data:
      1. Missing values get added to one branch of split
      2. Use classification error to determine where missing values go 
