# Goal
Build a classification model to predict whether or not a loan provided by [LendingClub](https://www.lendingclub.com/) is likely to default.
# File Description
- `.zip` files is data file.
  - `lending-club-data.csv.zip` (unzip `lending-club-data.csv`) consists of 122,607 people and 68 features
  - `module-5-assignment-1-train-idx.json.zip` 
  - `module-5-assignment-1-validation-idx.json.zip`
- `.ipynb` file is the solution of Week 3 program assignment 4
  - `Identifying safe loans with decision trees.ipynb`
- `.html` file is the html version of `.ipynb` file.
  - `Identifying+safe+loans+with+decision+trees.html`
- `.doc` file is the exported graphviz 
  - `simple_tree.dot.doc`
- `.png` file is the output image
  - `simple_tree.png`
# Snapshot
open `.html` file via brower for quick look.
# Algorithm
- Decision Tree
# Implementation
- Use Pandas to do some feature engineering.
- Train a decision-tree on the LendingClub dataset.
- Visualize the tree.
- Predict whether a loan will default along with prediction probabilities (on a validation set).
- Train a complex tree model and compare it to simple tree model.
# Implementation in detail
- Load the Lending Club dataset
- Exploring some features
  - print out the column names
- explore the distribution of the column safe_loans
- using a subset of features (categorical and numeric) 12 features
- Write a One-hot encoding function (67 features)
- Build a decision tree classifier with [sklearn.tree.DecisionTreeClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) denoted as small_model(max_depth=2) and decision_tree_model(max_depth=6)
- Visualizing a learned model with [graphviz](http://graphviz.readthedocs.org/en/latest/#)
- Making predictions
- Explore probability predictions
- Evaluating accuracy of the decision tree model
  - small_model 
    - training data: 0.61
    - validation data: 0.62
  - decision_tree_model
    - training data: 0.64
    - validation data 0.64
- Evaluating accuracy of a complex decision tree model big_model(max_depth=10)
  - big_model
    - training data: 0.66
    - validation data 0.63
  

  
