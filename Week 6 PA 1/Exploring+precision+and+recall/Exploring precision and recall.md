
# Exploring precision and recall

The goal of this second notebook is to understand precision-recall in the context of classifiers.

 * Use Amazon review data in its entirety.
 * Train a logistic regression model.
 * Explore various evaluation metrics: accuracy, confusion matrix, precision, recall.
 * Explore how various metrics can be combined to produce a cost of making an error.
 * Explore precision and recall curves.
 
Because we are using the full Amazon review dataset (not a subset of words or reviews), in this assignment we return to using GraphLab Create for its efficiency. As usual, let's start by **firing up GraphLab Create**.

Make sure you have the latest version of GraphLab Create (1.8.3 or later). If you don't find the decision tree module, then you would need to upgrade graphlab-create using

```
   pip install graphlab-create --upgrade
```
See [this page](https://dato.com/download/) for detailed instructions on upgrading.


```python
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
%matplotlib inline
```

# Load amazon review dataset


```python
products = pd.read_csv('amazon_baby.csv')
products.head(2)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>review</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Planetwise Flannel Wipes</td>
      <td>These flannel wipes are OK, but in my opinion ...</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Planetwise Wipe Pouch</td>
      <td>it came early and was not disappointed. i love...</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
print products.dtypes

print len(products)
```

    name      object
    review    object
    rating     int64
    dtype: object
    183531
    


```python

#products2 = products.fillna({'review':''})  # fill in N/A's in the review column

#print products2
print products[products['review'].isnull()].head(3)
print '\n'
print products.iloc[38]
```

                                                      name review  rating
    38        SoftPlay Twinkle Twinkle Elmo A Bedtime Book    NaN       5
    58                           Our Baby Girl Memory Book    NaN       5
    721  Summer Infant, Ultimate Training Pad - Twin Ma...    NaN       5
    
    
    name      SoftPlay Twinkle Twinkle Elmo A Bedtime Book
    review                                             NaN
    rating                                               5
    Name: 38, dtype: object
    

# Extract word counts and sentiments

As in the first assignment of this course, we compute the word counts for individual words and extract positive and negative sentiments from ratings. To summarize, we perform the following:

1. Remove punctuation.
2. Remove reviews with "neutral" sentiment (rating 3).
3. Set reviews with rating 4 or more to be positive and those with 2 or less to be negative.


```python
products = products.fillna({'review':''})  # fill in N/A's in the review column

def remove_punctuation(text):
    import string
    #if type(text) == float:
        #print text
    return text.translate(None, string.punctuation) 


products['review_clean'] = products['review'].apply(remove_punctuation)

products = products[products['rating'] != 3]

products['sentiment'] = products['rating'].apply(lambda rating : +1 if rating > 3 else -1)

```


```python
products.head(5)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>review</th>
      <th>rating</th>
      <th>review_clean</th>
      <th>sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Planetwise Wipe Pouch</td>
      <td>it came early and was not disappointed. i love...</td>
      <td>5</td>
      <td>it came early and was not disappointed i love ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Annas Dream Full Quilt with 2 Shams</td>
      <td>Very soft and comfortable and warmer than it l...</td>
      <td>5</td>
      <td>Very soft and comfortable and warmer than it l...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Stop Pacifier Sucking without tears with Thumb...</td>
      <td>This is a product well worth the purchase.  I ...</td>
      <td>5</td>
      <td>This is a product well worth the purchase  I h...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Stop Pacifier Sucking without tears with Thumb...</td>
      <td>All of my kids have cried non-stop when I trie...</td>
      <td>5</td>
      <td>All of my kids have cried nonstop when I tried...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Stop Pacifier Sucking without tears with Thumb...</td>
      <td>When the Binky Fairy came to our house, we did...</td>
      <td>5</td>
      <td>When the Binky Fairy came to our house we didn...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



Now, let's remember what the dataset looks like by taking a quick peek:


```python
products.shape
```




    (166752, 5)



## Split data into training and test sets

We split the data into a 80-20 split where 80% is in the training set and 20% is in the test set.


```python
with open('module-9-assignment-train-idx.json') as train_data_file:    
    train_data_idx = json.load(train_data_file)
with open('module-9-assignment-test-idx.json') as test_data_file:    
    test_data_idx = json.load(test_data_file)

print train_data_idx[:3]
print test_data_idx[:3]
```

    [0, 1, 2]
    [8, 9, 14]
    


```python
train_data = products.iloc[train_data_idx]
train_data.head(2)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>review</th>
      <th>rating</th>
      <th>review_clean</th>
      <th>sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Planetwise Wipe Pouch</td>
      <td>it came early and was not disappointed. i love...</td>
      <td>5</td>
      <td>it came early and was not disappointed i love ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Annas Dream Full Quilt with 2 Shams</td>
      <td>Very soft and comfortable and warmer than it l...</td>
      <td>5</td>
      <td>Very soft and comfortable and warmer than it l...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
print len(train_data[train_data['sentiment'] == 1])
print len(train_data[train_data['sentiment'] == -1])
print len(train_data)
```

    112164
    21252
    133416
    


```python
test_data = products.iloc[test_data_idx]
test_data.head(2)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>review</th>
      <th>rating</th>
      <th>review_clean</th>
      <th>sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9</th>
      <td>Baby Tracker&amp;reg; - Daily Childcare Journal, S...</td>
      <td>This has been an easy way for my nanny to reco...</td>
      <td>4</td>
      <td>This has been an easy way for my nanny to reco...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Baby Tracker&amp;reg; - Daily Childcare Journal, S...</td>
      <td>I love this journal and our nanny uses it ever...</td>
      <td>4</td>
      <td>I love this journal and our nanny uses it ever...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
print len(test_data[test_data['sentiment'] == 1])
print len(test_data[test_data['sentiment'] == -1])
print len(test_data)
```

    28095
    5241
    33336
    


```python
print len(train_data) + len(test_data)
```

    166752
    

## Build the word count vector for each review
We will now compute the word count for each word that appears in the reviews. A vector consisting of word counts is often referred to as bag-of-word features. Since most words occur in only a few reviews, word count vectors are sparse. For this reason, scikit-learn and many other tools use sparse matrices to store a collection of word count vectors. Refer to appropriate manuals to produce sparse word count vectors. General steps for extracting word count vectors are as follows:

- Learn a vocabulary (set of all words) from the training data. Only the words that show up in the training data will be considered for feature extraction.
- Compute the occurrences of the words in each review and collect them into a row vector.
- Build a sparse matrix where each row is the word count vector for the corresponding review. Call this matrix train_matrix.
- Using the same mapping between words and columns, convert the test data into a sparse matrix test_matrix.

The following cell uses CountVectorizer in scikit-learn. Notice the token_pattern argument in the constructor.


```python
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
     # Use this token pattern to keep single-letter words
# First, learn vocabulary from the training data and assign columns to words
# Then convert the training data into a sparse matrix
train_matrix = vectorizer.fit_transform(train_data['review_clean'])
# Second, convert the test data into a sparse matrix, using the same word-column mapping
test_matrix = vectorizer.transform(test_data['review_clean'])

```


```python
#print vectorizer.vocabulary_
```

## Train a logistic regression classifier

We will now train a logistic regression classifier with **sentiment** as the target and **word_count** as the features. We will set `validation_set=None` to make sure everyone gets exactly the same results.  

Remember, even though we now know how to implement logistic regression, we will use GraphLab Create for its efficiency at processing this Amazon dataset in its entirety.  The focus of this assignment is instead on the topic of precision and recall.


```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(train_matrix, train_data['sentiment'])
```




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)




```python
model.classes_
```




    array([-1,  1], dtype=int64)



# Model Evaluation

We will explore the advanced model evaluation concepts that were discussed in the lectures.

## Accuracy

One performance metric we will use for our more advanced exploration is accuracy, which we have seen many times in past assignments.  Recall that the accuracy is given by

$$
\mbox{accuracy} = \frac{\mbox{# correctly classified data points}}{\mbox{# total data points}}
$$

To obtain the accuracy of our trained models using GraphLab Create, simply pass the option `metric='accuracy'` to the `evaluate` function. We compute the **accuracy** of our logistic regression model on the **test_data** as follows:


```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_true=test_data['sentiment'].as_matrix(), y_pred=model.predict(test_matrix))
print "Test Accuracy: %s" % accuracy
```

    Test Accuracy: 0.932235421166
    

## Baseline: Majority class prediction

Recall from an earlier assignment that we used the **majority class classifier** as a baseline (i.e reference) model for a point of comparison with a more sophisticated classifier. The majority classifier model predicts the majority class for all data points. 

Typically, a good model should beat the majority class classifier. Since the majority class in this dataset is the positive class (i.e., there are more positive than negative reviews), the accuracy of the majority class classifier can be computed as follows:


```python
baseline = len(test_data[test_data['sentiment'] == 1]) / float(len(test_data))
print "Baseline accuracy (majority class classifier): %s" % baseline
```

    Baseline accuracy (majority class classifier): 0.842782577394
    

** Quiz Question:** Using accuracy as the evaluation metric, was our **logistic regression model** better than the baseline (majority class classifier)?

YES

- Test Accuracy: 0.932235421166
- Baseline accuracy (majority class classifier): 0.842782577394

## Confusion Matrix

The accuracy, while convenient, does not tell the whole story. For a fuller picture, we turn to the **confusion matrix**. In the case of binary classification, the confusion matrix is a 2-by-2 matrix laying out correct and incorrect predictions made in each label as follows:
```
              +---------------------------------------------+
              |                Predicted label              |
              +----------------------+----------------------+
              |          (+1)        |         (-1)         |
+-------+-----+----------------------+----------------------+
| True  |(+1) | # of true positives  | # of false negatives |
| label +-----+----------------------+----------------------+
|       |(-1) | # of false positives | # of true negatives  |
+-------+-----+----------------------+----------------------+
```
To print out the confusion matrix for a classifier, use `metric='confusion_matrix'`:


```python
from sklearn.metrics import confusion_matrix
cmat = confusion_matrix(y_true=test_data['sentiment'].as_matrix(),
                        y_pred=model.predict(test_matrix),
                        labels=model.classes_)    # use the same order of class as the LR model.
print ' target_label | predicted_label | count '
print '--------------+-----------------+-------'
# Print out the confusion matrix.
# NOTE: Your tool may arrange entries in a different order. Consult appropriate manuals.
for i, target_label in enumerate(model.classes_):
    for j, predicted_label in enumerate(model.classes_):
        print '{0:^13} | {1:^15} | {2:5d}'.format(target_label, predicted_label, cmat[i,j])

```

     target_label | predicted_label | count 
    --------------+-----------------+-------
         -1       |       -1        |  3790
         -1       |        1        |  1451
          1       |       -1        |   808
          1       |        1        | 27287
    

**Quiz Question**: How many predicted values in the **test set** are **false positives**?

1451

## Computing the cost of mistakes


Put yourself in the shoes of a manufacturer that sells a baby product on Amazon.com and you want to monitor your product's reviews in order to respond to complaints.  Even a few negative reviews may generate a lot of bad publicity about the product. So you don't want to miss any reviews with negative sentiments --- you'd rather put up with false alarms about potentially negative reviews instead of missing negative reviews entirely. In other words, **false positives cost more than false negatives**. (It may be the other way around for other scenarios, but let's stick with the manufacturer's scenario for now.)

Suppose you know the costs involved in each kind of mistake: 
1. \$100 for each false positive.
2. \$1 for each false negative.
3. Correctly classified reviews incur no cost.

**Quiz Question**: Given the stipulation, what is the cost associated with the logistic regression classifier's performance on the **test set**?


```python
FP = 1451
FN = 808
print 100*FP +1*FN
```

    145908
    

## Precision and Recall

You may not have exact dollar amounts for each kind of mistake. Instead, you may simply prefer to reduce the percentage of false positives to be less than, say, 3.5% of all positive predictions. This is where **precision** comes in:

$$
[\text{precision}] = \frac{[\text{# positive data points with positive predicitions}]}{\text{[# all data points with positive predictions]}} = \frac{[\text{# true positives}]}{[\text{# true positives}] + [\text{# false positives}]}
$$

So to keep the percentage of false positives below 3.5% of positive predictions, we must raise the precision to 96.5% or higher. 

**First**, let us compute the precision of the logistic regression classifier on the **test_data**.


```python
from sklearn.metrics import precision_score
precision = precision_score(y_true=test_data['sentiment'].as_matrix(), 
                            y_pred=model.predict(test_matrix))
print "Precision on test data: %s" % precision
```

    Precision on test data: 0.949509360429
    

**Quiz Question**: Out of all reviews in the **test set** that are predicted to be positive, what fraction of them are **false positives**? (Round to the second decimal place e.g. 0.25)


```python
print 1-precision
```

    0.0504906395713
    

**Quiz Question:** Based on what we learned in lecture, if we wanted to reduce this fraction of false positives to be below 3.5%, we would: (see the quiz)

A complementary metric is **recall**, which measures the ratio between the number of true positives and that of (ground-truth) positive reviews:

$$
[\text{recall}] = \frac{[\text{# positive data points with positive predicitions}]}{\text{[# all positive data points]}} = \frac{[\text{# true positives}]}{[\text{# true positives}] + [\text{# false negatives}]}
$$

Let us compute the recall on the **test_data**.


```python
from sklearn.metrics import recall_score
recall = recall_score(y_true=test_data['sentiment'].as_matrix(),
                      y_pred=model.predict(test_matrix))
print "Recall on test data: %s" % recall
```

    Recall on test data: 0.971240434241
    


```python
print 1
```

    1
    

**Quiz Question**: What fraction of the positive reviews in the **test_set** were correctly predicted as positive by the classifier?

**Quiz Question**: What is the recall value for a classifier that predicts **+1** for all data points in the **test_data**?

# Precision-recall tradeoff

In this part, we will explore the trade-off between precision and recall discussed in the lecture.  We first examine what happens when we use a different threshold value for making class predictions.  We then explore a range of threshold values and plot the associated precision-recall curve.  


## Varying the threshold

False positives are costly in our example, so we may want to be more conservative about making positive predictions. To achieve this, instead of thresholding class probabilities at 0.5, we can choose a higher threshold. 

Write a function called `apply_threshold` that accepts two things
* `probabilities` (an SArray of probability values)
* `threshold` (a float between 0 and 1).

The function should return an SArray, where each element is set to +1 or -1 depending whether the corresponding probability exceeds `threshold`.


```python
def apply_threshold(probabilities, threshold):
    ### YOUR CODE GOES HERE
    # +1 if >= threshold and -1 otherwise.
    result = np.ones(len(probabilities))
    result[probabilities < threshold] = -1
    
    return result
```

Run prediction with `output_type='probability'` to get the list of probability values. Then use thresholds set at 0.5 (default) and 0.9 to make predictions from these probability values.


```python
probabilities = model.predict_proba(test_matrix)[:,1]

predictions_with_default_threshold = apply_threshold(probabilities, 0.5)

predictions_with_high_threshold = apply_threshold(probabilities, 0.9)

```


```python

print predictions_with_default_threshold
print predictions_with_high_threshold
print '\n'
print sum(probabilities >= 0.5)
print sum(probabilities >= 0.9)
print '\n'
print predictions_with_default_threshold * (predictions_with_default_threshold==-1)
print len(predictions_with_default_threshold * (predictions_with_default_threshold==-1))
print '\n'
print np.sum(predictions_with_default_threshold >0)
print np.sum(predictions_with_high_threshold>0)

```

    [ 1.  1.  1. ...,  1.  1.  1.]
    [-1.  1.  1. ...,  1.  1.  1.]
    
    
    28738
    25069
    
    
    [ 0.  0.  0. ...,  0.  0.  0.]
    33336
    
    
    28738
    25069
    


```python
print "Number of positive predicted reviews (threshold = 0.5): %s" % (predictions_with_default_threshold == 1).sum()
```

    Number of positive predicted reviews (threshold = 0.5): 28738
    


```python
print "Number of positive predicted reviews (threshold = 0.9): %s" % (predictions_with_high_threshold == 1).sum()
```

    Number of positive predicted reviews (threshold = 0.9): 25069
    

**Quiz Question**: What happens to the number of positive predicted reviews as the threshold increased from 0.5 to 0.9?

## Exploring the associated precision and recall as the threshold varies

By changing the probability threshold, it is possible to influence precision and recall. We can explore this as follows:


```python
# Threshold = 0.5
precision_with_default_threshold = precision_score(y_true=test_data['sentiment'].as_matrix(), y_pred=predictions_with_default_threshold)

recall_with_default_threshold = recall_score(y_true=test_data['sentiment'].as_matrix(), y_pred=predictions_with_default_threshold)

# Threshold = 0.9
precision_with_high_threshold = precision_score(y_true=test_data['sentiment'].as_matrix(), y_pred=predictions_with_high_threshold)

recall_with_high_threshold = recall_score(y_true=test_data['sentiment'].as_matrix(), y_pred=predictions_with_high_threshold)

                                      
```


```python
print "Precision (threshold = 0.5): %s" % precision_with_default_threshold
print "Recall (threshold = 0.5)   : %s" % recall_with_default_threshold
```

    Precision (threshold = 0.5): 0.949509360429
    Recall (threshold = 0.5)   : 0.971240434241
    


```python
print "Precision (threshold = 0.9): %s" % precision_with_high_threshold
print "Recall (threshold = 0.9)   : %s" % recall_with_high_threshold
```

    Precision (threshold = 0.9): 0.981570864414
    Recall (threshold = 0.9)   : 0.875849795337
    

**Quiz Question (variant 1)**: Does the **precision** increase with a higher threshold?

**Quiz Question (variant 2)**: Does the **recall** increase with a higher threshold?

## Precision-recall curve

Now, we will explore various different values of tresholds, compute the precision and recall scores, and then plot the precision-recall curve.


```python
threshold_values = np.linspace(0.5, 1, num=100)
print threshold_values
```

    [ 0.5         0.50505051  0.51010101  0.51515152  0.52020202  0.52525253
      0.53030303  0.53535354  0.54040404  0.54545455  0.55050505  0.55555556
      0.56060606  0.56565657  0.57070707  0.57575758  0.58080808  0.58585859
      0.59090909  0.5959596   0.6010101   0.60606061  0.61111111  0.61616162
      0.62121212  0.62626263  0.63131313  0.63636364  0.64141414  0.64646465
      0.65151515  0.65656566  0.66161616  0.66666667  0.67171717  0.67676768
      0.68181818  0.68686869  0.69191919  0.6969697   0.7020202   0.70707071
      0.71212121  0.71717172  0.72222222  0.72727273  0.73232323  0.73737374
      0.74242424  0.74747475  0.75252525  0.75757576  0.76262626  0.76767677
      0.77272727  0.77777778  0.78282828  0.78787879  0.79292929  0.7979798
      0.8030303   0.80808081  0.81313131  0.81818182  0.82323232  0.82828283
      0.83333333  0.83838384  0.84343434  0.84848485  0.85353535  0.85858586
      0.86363636  0.86868687  0.87373737  0.87878788  0.88383838  0.88888889
      0.89393939  0.8989899   0.9040404   0.90909091  0.91414141  0.91919192
      0.92424242  0.92929293  0.93434343  0.93939394  0.94444444  0.94949495
      0.95454545  0.95959596  0.96464646  0.96969697  0.97474747  0.97979798
      0.98484848  0.98989899  0.99494949  1.        ]
    

For each of the values of threshold, we compute the precision and recall scores.


```python
precision_all = []
recall_all = []

probabilities = model.predict_proba(test_matrix)[:,1]
for threshold in threshold_values:
    predictions = apply_threshold(probabilities, threshold)
    
    precision = precision_score(y_true=test_data['sentiment'].as_matrix(), y_pred=predictions)
    recall = recall_score(y_true=test_data['sentiment'].as_matrix(), y_pred=predictions)
    
    precision_all.append(precision)
    recall_all.append(recall)
```

Now, let's plot the precision-recall curve to visualize the precision-recall tradeoff as we vary the threshold.


```python
import matplotlib.pyplot as plt
%matplotlib inline

def plot_pr_curve(precision, recall, title):
    plt.rcParams['figure.figsize'] = 7, 5
    plt.locator_params(axis = 'x', nbins = 5)
    plt.plot(precision, recall, 'b-', linewidth=4.0, color = '#B0017F')
    plt.title(title)
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.rcParams.update({'font.size': 16})
    
plot_pr_curve(precision_all, recall_all, 'Precision recall curve (all)')
```


![png](output_69_0.png)


**Quiz Question**: Among all the threshold values tried, what is the **smallest** threshold value that achieves a precision of 96.5% or better? Round your answer to 3 decimal places.


```python
print np.array(threshold_values)[np.array(precision_all) >= 0.965]
```

    [ 0.70707071  0.71212121  0.71717172  0.72222222  0.72727273  0.73232323
      0.73737374  0.74242424  0.74747475  0.75252525  0.75757576  0.76262626
      0.76767677  0.77272727  0.77777778  0.78282828  0.78787879  0.79292929
      0.7979798   0.8030303   0.80808081  0.81313131  0.81818182  0.82323232
      0.82828283  0.83333333  0.83838384  0.84343434  0.84848485  0.85353535
      0.85858586  0.86363636  0.86868687  0.87373737  0.87878788  0.88383838
      0.88888889  0.89393939  0.8989899   0.9040404   0.90909091  0.91414141
      0.91919192  0.92424242  0.92929293  0.93434343  0.93939394  0.94444444
      0.94949495  0.95454545  0.95959596  0.96464646  0.96969697  0.97474747
      0.97979798  0.98484848  0.98989899  0.99494949  1.        ]
    

**Quiz Question**: Using `threshold` = 0.98, how many **false negatives** do we get on the **test_data**? (**Hint**: You may use the `graphlab.evaluation.confusion_matrix` function implemented in GraphLab Create.)


```python
predictions_with_098_threshold = apply_threshold(probabilities, 0.98)
sth = (np.array(test_data['sentiment'].as_matrix()) > 0) * (predictions_with_098_threshold < 0)
print sum(sth)
```

    8239
    


```python
cmat_098 = confusion_matrix(y_true=test_data['sentiment'].as_matrix(),
                        y_pred=predictions_with_098_threshold,
                        labels=model.classes_)    # use the same order of class as the LR model.
print ' target_label | predicted_label | count '
print '--------------+-----------------+-------'
# Print out the confusion matrix.
# NOTE: Your tool may arrange entries in a different order. Consult appropriate manuals.
for i, target_label in enumerate(model.classes_):
    for j, predicted_label in enumerate(model.classes_):
        print '{0:^13} | {1:^15} | {2:5d}'.format(target_label, predicted_label, cmat_098[i,j])
```

     target_label | predicted_label | count 
    --------------+-----------------+-------
         -1       |       -1        |  5048
         -1       |        1        |   193
          1       |       -1        |  8239
          1       |        1        | 19856
    

This is the number of false negatives (i.e the number of reviews to look at when not needed) that we have to deal with using this classifier.

# Evaluating specific search terms

So far, we looked at the number of false positives for the **entire test set**. In this section, let's select reviews using a specific search term and optimize the precision on these reviews only. After all, a manufacturer would be interested in tuning the false positive rate just for their products (the reviews they want to read) rather than that of the entire set of products on Amazon.

## Precision-Recall on all baby related items

From the **test set**, select all the reviews for all products with the word 'baby' in them.


```python
baby_reviews =  test_data[test_data['name'].apply(lambda x: 'baby' in str(x).lower())]
```

Now, let's predict the probability of classifying these reviews as positive:


```python
baby_matrix = vectorizer.transform(baby_reviews['review_clean'])
probabilities = model.predict_proba(baby_matrix)[:,1]
```

Let's plot the precision-recall curve for the **baby_reviews** dataset.

**First**, let's consider the following `threshold_values` ranging from 0.5 to 1:


```python
threshold_values = np.linspace(0.5, 1, num=100)
```

**Second**, as we did above, let's compute precision and recall for each value in `threshold_values` on the **baby_reviews** dataset.  Complete the code block below.


```python
precision_all = []
recall_all = []

for threshold in threshold_values:
    
    # Make predictions. Use the `apply_threshold` function 
    ## YOUR CODE HERE 
    predictions = apply_threshold(probabilities, threshold)

    # Calculate the precision.
    # YOUR CODE HERE
    precision = precision_score(y_true=baby_reviews['sentiment'].as_matrix(), y_pred=predictions)
    
    # YOUR CODE HERE
    recall = recall_score(y_true=baby_reviews['sentiment'].as_matrix(), y_pred=predictions)
    
    # Append the precision and recall scores.
    precision_all.append(precision)
    recall_all.append(recall)  
```


```python
plot_pr_curve(precision_all, recall_all, "Precision-Recall (Baby)")
```


![png](output_85_0.png)


**Quiz Question**: Among all the threshold values tried, what is the **smallest** threshold value that achieves a precision of 96.5% or better for the reviews of data in **baby_reviews**? Round your answer to 3 decimal places.


```python
print np.array(threshold_values)[np.array(precision_all) >= 0.965]
```

    [ 0.73232323  0.73737374  0.74242424  0.74747475  0.75252525  0.75757576
      0.76262626  0.76767677  0.77272727  0.77777778  0.78282828  0.78787879
      0.79292929  0.7979798   0.8030303   0.80808081  0.81313131  0.81818182
      0.82323232  0.82828283  0.83333333  0.83838384  0.84343434  0.84848485
      0.85353535  0.85858586  0.86363636  0.86868687  0.87373737  0.87878788
      0.88383838  0.88888889  0.89393939  0.8989899   0.9040404   0.90909091
      0.91414141  0.91919192  0.92424242  0.92929293  0.93434343  0.93939394
      0.94444444  0.94949495  0.95454545  0.95959596  0.96464646  0.96969697
      0.97474747  0.97979798  0.98484848  0.98989899  0.99494949  1.        ]
    

larger

**Quiz Question:** Is this threshold value smaller or larger than the threshold used for the entire dataset to achieve the same specified precision of 96.5%?

**Finally**, let's plot the precision recall curve.


```python
plot_pr_curve(precision_all, recall_all, "Precision-Recall (Baby)")
```
