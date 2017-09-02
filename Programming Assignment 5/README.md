# Goal
Implement your own binary decision tree classifier.
# File Description
- `.zip` files is data file.
  - `[lending-club-data.csv.zip](implement your own binary decision tree classifier)` (unzip `lending-club-data.csv`) consists of 122,607 people and 68 features
  - `module-5-assignment-2-train-idx.json.zip` 
  - `module-5-assignment-2-test-idx.json.zip`
- `.ipynb` file is the solution of Week 3 program assignment 5
  - `Implementing binary decision trees.ipynb`
- `.html` file is the html version of `.ipynb` file.
  - `Implementing+binary+decision+trees.html`
# Snapshot
open `.html` file via brower for quick look.
# Algorithm
- Decision Tree
# Implementation
- Use Pandas to do some feature engineering.
- Transform categorical variables into binary variables.
- Write a function to compute the number of misclassified examples in an intermediate node.
- Write a function to find the best feature to split on.
- Build a binary decision tree from scratch.
- Make predictions using the decision tree.
- Evaluate the accuracy of the decision tree.
- Visualize the decision at the root node.
**Important Note**: In this assignment, we will focus on building decision trees where the data contain only binary (0 or 1) features. This allows us to avoid dealing with:
- Multiple intermediate nodes in a split
- The thresholding issues of real-valued features.
# Implementation in detail
- Transform categorical data into binary features
- Write a function to count number of mistakes while predicting majority class
- Write a function to pick best feature to split on
- Write a function that creates a leaf node given a set of target values
- Building the tree
- Making predictions with a decision tree
- Write a function evaluating your decision tree
- Printing out a decision stump
- Exploring the intermediate left subtree
  

  
