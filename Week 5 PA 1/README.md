# Goal
explore the use of boosting with the pre-implemented gradient boosted trees
# File Description
- `.zip` file is data file.
  - [lending-club-data.csv.zip](https://github.com/SSQ/Coursera-UW-Machine-Learning-Classification/blob/master/Programming%20Assignment%204/lending-club-data.csv.zip) (unzip `lending-club-data.csv`) consists of 122,607 people and 68 features
- `.json` files are idx file
  - `module-8-assignment-1-train-idx.json` 
  - `module-8-assignment-1-validation-idx.json`
- `.ipynb` file is the solution of Week 4 program assignment 6
  - `module-8-boosting-assignment-1.ipynb`
- `.html` file is the html version of `.ipynb` file.
  - `module-8-boosting-assignment-1.html`
# Snapshot
- **Recommend** open `md` file inside a file
- open `.html` file via brower for quick look.
# Algorithm
- gradient boosted trees
# Implementation
- Use Pandas to do some feature engineering.
- Train a boosted ensemble of decision-trees (gradient boosted trees) on the lending club dataset.
- Predict whether a loan will default along with prediction probabilities (on a validation set).
- Evaluate the trained model and compare it with a baseline.
- Find the most positive and negative loans using the learned model.
- Explore how the number of trees influences classification performance.
# Implementation in detail
- Select 24 features
- Transform categorical data into binary features
- Create Gradient boosted tree classifier with the built-in scikit learn gradient boosting classifier ([sklearn.ensemble.GradientBoostingClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html))
  - max_depth=6 
  - n_estimators=5
- Evaluating the model on the validation data **accuracy: 0.66**
- Comparison with decision trees
- Effect of adding more trees with 10, 50, 100, 200, and 500 trees. **validation accuracy:**
  - 0.666307626023
  - 0.685264971995
  - **0.690219732874 model_100**
  - 0.68935803533
  - 0.687203791469
- Plot the training and validation error vs. number of trees 

  
