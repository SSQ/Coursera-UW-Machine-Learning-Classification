# Goal
- Using product review data from Amazon.com to predict whether the sentiments about a product (from its reviews) are positive or negative.
# File Description
- `.rar` files is data file.
  - `amazon_baby.rar` (unzip `amazon_baby.csv`) consists of 183,531 customers with `name`, `review`, `rating`.
- `.ipynb` file is the solution of Week 1 program assignment
  - `Predicting sentiment from product reviews with pandas.ipynb`
- `.html` file is the html version of `.ipynb` file.
  - `Predicting+sentiment+from+product+reviews+with+pandas.html`
# Snapshot
**Recommend** open `md` file
open `.html` file via brower for quick look.
# Algorithm
- Logistic Regression
# Implementation
- Use Pandas to do some feature engineering
- Train a **logistic regression** model to predict the sentiment of product reviews.
- Inspect the weights (coefficients) of a trained logistic regression model.
- Make a prediction (both class and probability) of sentiment for a new product review.
- Given the logistic regression weights, predictors and ground truth labels, write a function to compute the accuracy of the model.
- Inspect the coefficients of the logistic regression model and interpret their meanings.
- Compare multiple logistic regression models.
# Implementation in detail
- Load Amazon dataset. `pd.read_csv()`
- Perform text cleaning. removing punctuation, fill in N/A's in the empty review (pandas.fillna({'review':''}))
- Extract Sentiments.
- Build the word count vector for each review. [CountVectorizer](http://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)
- Train a sentiment classifier with logistic regression. [LogisticRegression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- Making predictions with logistic regression.
- Prediciting Sentiment.
- Probability Predictions.
- Find the most positive (and negative) review.
- Compute accuracy of the classifier.
- Learn another classifier with fewer words.
- Train a logistic regression model on a subset of data.
- Comparing models.
- Baseline: Majority class prediction.

