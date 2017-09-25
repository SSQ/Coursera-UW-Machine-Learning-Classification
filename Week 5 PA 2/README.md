# Goal
Implementing your own Adaboost
# File Description
- `.zip` file is data file.
  - [lending-club-data.csv.zip](https://github.com/SSQ/Coursera-UW-Machine-Learning-Classification/blob/master/Programming%20Assignment%204/lending-club-data.csv.zip) (unzip `lending-club-data.csv`) consists of 122,607 people and 68 features
- `.json` files are idx file
  - `module-8-assignment-2-train-idx.json` 
  - `module-8-assignment-2-test-idx.json`
- `.ipynb` file is the solution of Week 5 program assignment 2
  - `module-8-boosting-assignment-2.ipynb`
- `.html` file is the html version of `.ipynb` file.
  - `module-8-boosting-assignment-2.html`
# Snapshot
open `.html` file via brower for quick look.
# Algorithm
- Implementing your own Adaboost
# Implementation
- Use Pandas to do some feature engineering.
- Train a boosted ensemble of decision-trees (gradient boosted trees) on the lending club dataset.
- Predict whether a loan will default along with prediction probabilities (on a validation set).
- Evaluate the trained model and compare it with a baseline.
- Find the most positive and negative loans using the learned model.
- Explore how the number of trees influences classification performance.
# Implementation in detail
- Select 4 features 
- Transform categorical data into 25 binary features
- Write a function `weighted_decision_tree_create(data, features, target, data_weights, current_depth = 1, max_depth = 10)` to create Weighted decision trees
  - Write a function `intermediate_node_weighted_mistakes(labels_in_node, data_weights)` to compute weight of mistakes. return the lower of the two weights of mistakes, along with the class associated with that weight
  - Write a function `best_splitting_feature(data, features, target, data_weights)` to pick best feature to split on
- Building the tree `small_data_decision_tree = weighted_decision_tree_create(train_data, features, target, example_data_weights, max_depth=2)`
- Making predictions with a weighted decision tree `classify(tree, x, annotate = False)`
- Evaluating the tree `evaluate_classification_error(tree, data)`
  - `evaluate_classification_error(small_data_decision_tree, train_data)` **0.400**
  - `evaluate_classification_error(small_data_decision_tree, test_data)` **0.398**
- Example: Training a weighted decision tree `small_data_decision_tree_subset_20 = weighted_decision_tree_create(train_data, features, target, example_data_weights, max_depth=2)`
- Evaluating the tree `small_data_decision_tree_subset_20 = weighted_decision_tree_create(train_data, features, target, example_data_weights, max_depth=2)`
  - `evaluate_classification_error(small_data_decision_tree_subset_20, subset_20)` **0.050**
  - `evaluate_classification_error(small_data_decision_tree_subset_20, train_data)` **0.48**
- Write a function `adaboost_with_tree_stumps(data, features, target, num_tree_stumps)` to implement your own Adaboost (on decision stumps). `return weights, tree_stumps`
- Write a function `stump_weights, tree_stumps = adaboost_with_tree_stumps(train_data, features, target, num_tree_stumps=10)` to train a boosted ensemble of 10 stumps
- Write a function `predict_adaboost(stump_weights, tree_stumps, data)` to making predictions
  - `predict_adaboost(stump_weights, tree_stumps, train_data)` **0.615516870836**
  - `predictions = predict_adaboost(stump_weights, tree_stumps, test_data)` **0.620314519604**
- Write a function `stump_weights, tree_stumps = adaboost_with_tree_stumps(train_data, features, target, num_tree_stumps=30)` to train a boosted ensemble of 30 stumps
  - Evaluation on the train data **0.378734150011**
  - Evaluation on the test data **0.376777251185**



  
