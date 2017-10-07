# Goal
- Understand precision-recall in the context of classifiers.
# File Description
- `.rar` files is data file.
  - [`amazon_baby.rar`](https://github.com/SSQ/Coursera-UW-Machine-Learning-Classification/blob/master/Programming%20Assignment%201/amazon_baby.rar) (unzip `amazon_baby.csv`) consists of 183,531 customers with `name`, `review`, `rating`.
- `.ipynb` file is the solution of Week 6 program assignment
  - `Exploring precision and recall.ipynb`
- `.html` file is the html version of `.ipynb` file.
  - `Exploring+precision+and+recall.html`
# Snapshot
open `.html` file via brower for quick look.
# Algorithm
- precision-recall 
# Implementation
- Use Amazon review data in its entirety.
- Train a logistic regression model.
- Explore various evaluation metrics: accuracy, confusion matrix, precision, recall.
- Explore how various metrics can be combined to produce a cost of making an error.
- Explore precision and recall curves.
# Implementation in detail
- Load Amazon dataset. `pd.read_csv()`
- Perform text cleaning. removing punctuation, fill in N/A's in the empty review (pandas.fillna({'review':''}))
- Extract Sentiments.
- Build the word count vector for each review. [CountVectorizer](http://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)
- Train a sentiment classifier with logistic regression. [LogisticRegression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- Making predictions with logistic regression.
- Model Evaluation
  - Accuracy. `from sklearn.metrics import accuracy_score`
  - Baseline: Majority class prediction. 
  - Confusion Matrix. `from sklearn.metrics import confusion_matrix`
  - Precision and Recall. `from sklearn.metrics import precision_score` `from sklearn.metrics import recall_score`
- Precision-recall tradeoff
  - Varying the threshold. Write a function `apply_threshold(probabilities, threshold)`
  - Exploring the associated precision and recall as the threshold varies
  - Precision-recall curve
- Evaluating specific search terms
  - Precision-Recall on all baby related items

