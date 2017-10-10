# Goal
Explore various techniques for preventing overfitting in decision trees.
# File Description
- `.zip` file is data file.
  - [lending-club-data.csv.zip](https://github.com/SSQ/Coursera-UW-Machine-Learning-Classification/blob/master/Programming%20Assignment%204/lending-club-data.csv.zip) (unzip `lending-club-data.csv`) consists of 122,607 people and 68 features
- `.json` files are idx file
  - `module-6-assignment-train-idx.json` 
  - `module-6-assignment-validation-idx.json`
- `.ipynb` file is the solution of Week 4 program assignment 6
  - `Decision Trees in Practice.ipynb`
- `.html` file is the html version of `.ipynb` file.
  - `Decision+Trees+in+Practice.html`
# Snapshot
- **Recommend** open `md` file
- open `.html` file via brower for quick look.
# Algorithm
- Preventing overfitting in decision trees
# Implementation
- Implement binary decision trees with different early stopping methods.
- Compare models with different stopping parameters.
- Visualize the concept of overfitting in decision trees.
# Implementation in detail
- Transform categorical data into binary features
- Write a function `reached_minimum_node_size` for Early stopping condition 2: Minimum node size
- Write a function `error_reduction` for Early stopping condition 3: Minimum gain in error reduction
- Update function `decision_tree_create` with adding 3 Early stopping conditions
- Building the tree with adding 3 Early stopping conditions
  - `my_decision_tree_new` with `min_error_reduction=0.0` classification error validation data: 0.3836
  - `my_decision_tree_old` with `min_error_reduction=-1` classification error validation data: 0.3837
- Exploring the effect of max_depth
  - set min_node_size = 0 and min_error_reduction = -1
    - model_1: max_depth = 2 (too small)
    - model_2: max_depth = 6 (just right)
    - model_3: max_depth = 14 (may be too large)
  - Evaluating the models
    - Training data, classification error 
      - (model 1): 0.400037610144
      - (model 2): 0.381850419084
      - (model 3): 0.374462712229
    - Validation data, classification error 
      - (model 1): 0.398104265403
      - (model 2): 0.383778543731
      - (model 3): 0.380008616975
  - Measuring the complexity of the tree
    - number of leaves 
      - in model_1 is : 4
      - in model_2 is : 41
      - in model_3 is : 341
- Exploring the effect of min_error
  - set max_depth = 6, and min_node_size = 0.
    - model_4: min_error_reduction = -1 (ignoring this early stopping condition)
    - model_5: min_error_reduction = 0 (just right)
    - model_6: min_error_reduction = 5 (too positive)
  - Validation data, classification error 
    - (model 4): 0.383778543731
    - (model 5): 0.383778543731
    - (model 6): 0.503446790177
  - Measuring the complexity of the tree
    - number of leaves 
      - in model_4 is : 41
      - in model_5 is : 13
      - in model_6 is : 1
- Exploring the effect of min_node_size
  - set max_depth = 6, and min_error_reduction = -1.
    - model_7: min_node_size = 0 (too small)
    - model_8: min_node_size = 2000 (just right)
    - model_9: min_node_size = 50000 (too large)
  - Validation data, classification error 
    - (model 7): 0.383778543731
    - (model 8): 0.384532529082
    - (model 9): 0.503446790177
  - Measuring the complexity of the tree
    - number of leaves 
      - in model_7 is : 41
      - in model_8 is : 19
      - in model_9 is : 1

  

  
