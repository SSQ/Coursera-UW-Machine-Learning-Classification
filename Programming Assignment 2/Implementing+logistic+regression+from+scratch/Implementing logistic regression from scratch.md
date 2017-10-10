

```python
import pandas as pd
import numpy as np
```

## Load review dataset


```python
products = pd.read_csv('amazon_baby_subset.csv')
```

### 1. listing the name of the first 10 products in the dataset.


```python
products['name'][:10]
```




    0    Stop Pacifier Sucking without tears with Thumb...
    1      Nature's Lullabies Second Year Sticker Calendar
    2      Nature's Lullabies Second Year Sticker Calendar
    3                          Lamaze Peekaboo, I Love You
    4    SoftPlay Peek-A-Boo Where's Elmo A Children's ...
    5                            Our Baby Girl Memory Book
    6    Hunnt&reg; Falling Flowers and Birds Kids Nurs...
    7    Blessed By Pope Benedict XVI Divine Mercy Full...
    8    Cloth Diaper Pins Stainless Steel Traditional ...
    9    Cloth Diaper Pins Stainless Steel Traditional ...
    Name: name, dtype: object



### 2. counting the number of positive and negative reviews.


```python
print (products['sentiment'] == 1).sum()
print (products['sentiment'] == -1).sum()
print (products['sentiment']).count()
```

    26579
    26493
    53072
    

## Apply text cleaning on the review data

### 3. load the features


```python
import json
with open('important_words.json') as important_words_file:    
    important_words = json.load(important_words_file)
print important_words[:3]
```

    [u'baby', u'one', u'great']
    

### 4. data transformations:
- fill n/a values in the review column with empty strings
- Remove punctuation
- Compute word counts (only for important_words)


```python
products = products.fillna({'review':''})  # fill in N/A's in the review column

def remove_punctuation(text):
    import string
    return text.translate(None, string.punctuation) 

products['review_clean'] = products['review'].apply(remove_punctuation)
products.head(3)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>review</th>
      <th>rating</th>
      <th>sentiment</th>
      <th>review_clean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Stop Pacifier Sucking without tears with Thumb...</td>
      <td>All of my kids have cried non-stop when I trie...</td>
      <td>5</td>
      <td>1</td>
      <td>All of my kids have cried nonstop when I tried...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Nature's Lullabies Second Year Sticker Calendar</td>
      <td>We wanted to get something to keep track of ou...</td>
      <td>5</td>
      <td>1</td>
      <td>We wanted to get something to keep track of ou...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Nature's Lullabies Second Year Sticker Calendar</td>
      <td>My daughter had her 1st baby over a year ago. ...</td>
      <td>5</td>
      <td>1</td>
      <td>My daughter had her 1st baby over a year ago S...</td>
    </tr>
  </tbody>
</table>
</div>



### 5. compute a count for the number of times the word occurs in the review


```python
for word in important_words:
    products[word] = products['review_clean'].apply(lambda s : s.split().count(word))
```


```python
products.head(1)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>review</th>
      <th>rating</th>
      <th>sentiment</th>
      <th>review_clean</th>
      <th>baby</th>
      <th>one</th>
      <th>great</th>
      <th>love</th>
      <th>use</th>
      <th>...</th>
      <th>seems</th>
      <th>picture</th>
      <th>completely</th>
      <th>wish</th>
      <th>buying</th>
      <th>babies</th>
      <th>won</th>
      <th>tub</th>
      <th>almost</th>
      <th>either</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Stop Pacifier Sucking without tears with Thumb...</td>
      <td>All of my kids have cried non-stop when I trie...</td>
      <td>5</td>
      <td>1</td>
      <td>All of my kids have cried nonstop when I tried...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 198 columns</p>
</div>



 ### 7. compute the number of product reviews that contain the word perfect.


```python
products['contains_perfect'] = products['perfect'] >=1
print products['contains_perfect'].sum()
```

    2955
    

## 1. Quiz Question. 
How many reviews contain the word perfect?

## Answer 
2955

## Convert data frame to multi-dimensional array

### 8.  convert our data frame to a multi-dimensional array.
The function should accept three parameters:
- dataframe: a data frame to be converted
- features: a list of string, containing the names of the columns that are used as features.
- label: a string, containing the name of the single column that is used as class labels.

The function should return two values:

- one 2D array for features
- one 1D array for class labels

The function should do the following:
- Prepend a new column constant to dataframe and fill it with 1's. This column takes account of the intercept term. Make sure that the constant column appears first in the data frame.
- Prepend a string 'constant' to the list features. Make sure the string 'constant' appears first in the list.
- Extract columns in dataframe whose names appear in the list features.
- Convert the extracted columns into a 2D array using a function in the data frame library. If you are using Pandas, you would use as_matrix() function.
- Extract the single column in dataframe whose name corresponds to the string label.
- Convert the column into a 1D array.
- Return the 2D array and the 1D array.


```python
def get_numpy_data(dataframe, features, label):
    dataframe['constant'] = 1
    features = ['constant'] + features
    features_frame = dataframe[features]
    feature_matrix = features_frame.as_matrix()
    label_sarray = dataframe[label]
    label_array = label_sarray.as_matrix()
    return(feature_matrix, label_array)
```

### 9. extract two arrays feature_matrix and sentiment


```python
feature_matrix, sentiment = get_numpy_data(products, important_words, 'sentiment')
```

## 2. Quiz Question: 
How many features are there in the feature_matrix?


```python
print feature_matrix.shape
```

    (53072L, 194L)
    

## 2. Answer:
194

## 3. Quiz Question: 
Assuming that the intercept is present, how does the number of features in feature_matrix relate to the number of features in the logistic regression model?

## Estimating conditional probability with link function

### 10. Compute predictions given by the link function.
- Take two parameters: feature_matrix and coefficients.
- First compute the dot product of feature_matrix and coefficients.
- Then compute the link function P(y=+1|x,w).
- Return the predictions given by the link function.


```python
'''
feature_matrix: N * D
coefficients: D * 1
predictions: N * 1
produces probablistic estimate for P(y_i = +1 | x_i, w).
estimate ranges between 0 and 1.
'''

def predict_probability(feature_matrix, coefficients):
    # Take dot product of feature_matrix and coefficients  
    # YOUR CODE HERE
    score = np.dot(feature_matrix, coefficients) # N * 1
    
    # Compute P(y_i = +1 | x_i, w) using the link function
    # YOUR CODE HERE
    predictions = 1.0/(1+np.exp(-score))
    
    # return predictions
    return predictions
```

## Compute derivative of log likelihood with respect to a single coefficient

### 11. computes the derivative of log likelihood with respect to a single coefficient w_j
The function should do the following:

- Take two parameters errors and feature.
- Compute the dot product of errors and feature.
- Return the dot product. This is the derivative with respect to a single coefficient w_j.


```python
"""
errors: N * 1
feature: N * 1
derivative: 1 
"""
def feature_derivative(errors, feature):     
    # Compute the dot product of errors and feature
    derivative = np.dot(np.transpose(errors), feature)
    # Return the derivative
    return derivative
```

### 12. Write a function compute_log_likelihood


```python
def compute_log_likelihood(feature_matrix, sentiment, coefficients):
    indicator = (sentiment==+1)
    scores = np.dot(feature_matrix, coefficients)
    # scores.shape (53072L, 1L)
    # indicator.shape (53072L,)
    lp = np.sum((np.transpose(np.array([indicator]))-1)*scores - np.log(1. + np.exp(-scores)))
    return lp
```

## Taking gradient steps

### 13. Write a function logistic_regression to fit a logistic regression model using gradient ascent.
The function accepts the following parameters:

- feature_matrix: 2D array of features
- sentiment: 1D array of class labels
- initial_coefficients: 1D array containing initial values of coefficients
- step_size: a parameter controlling the size of the gradient steps
- max_iter: number of iterations to run gradient ascent
- The function returns the last set of coefficients after performing gradient ascent.

The function carries out the following steps:

1. Initialize vector coefficients to initial_coefficients.
1. Predict the class probability P(yi=+1|xi,w) using your predict_probability function and save it to variable predictions.
1. Compute indicator value for (yi=+1) by comparing sentiment against +1. Save it to variable indicator.
1. Compute the errors as difference between indicator and predictions. Save the errors to variable errors.
1. For each j-th coefficient, compute the per-coefficient derivative by calling feature_derivative with the j-th column of feature_matrix. Then increment the j-th coefficient by (step_size*derivative).
1. Once in a while, insert code to print out the log likelihood.
1. Repeat steps 2-6 for max_iter times.


```python
# coefficients: D * 1
from math import sqrt
def logistic_regression(feature_matrix, sentiment, initial_coefficients, step_size, max_iter):
    coefficients = np.array(initial_coefficients) # make sure it's a numpy array
    # lplist = []
    for itr in xrange(max_iter):
        # Predict P(y_i = +1|x_1,w) using your predict_probability() function
        # YOUR CODE HERE
        predictions = predict_probability(feature_matrix, coefficients)

        # Compute indicator value for (y_i = +1)
        indicator = (sentiment==+1)

        # Compute the errors as indicator - predictions
        errors = np.transpose(np.array([indicator])) - predictions

        for j in xrange(len(coefficients)): # loop over each coefficient
            # Recall that feature_matrix[:,j] is the feature column associated with coefficients[j]
            # compute the derivative for coefficients[j]. Save it in a variable called derivative
            # YOUR CODE HERE
            derivative = feature_derivative(errors, feature_matrix[:,j])

            # add the step size times the derivative to the current coefficient
            # YOUR CODE HERE
            coefficients[j] += step_size*derivative

        # Checking whether log likelihood is increasing
        if itr <= 15 or (itr <= 100 and itr % 10 == 0) or (itr <= 1000 and itr % 100 == 0) \
        or (itr <= 10000 and itr % 1000 == 0) or itr % 10000 == 0:
            # lplist.append(compute_log_likelihood(feature_matrix, sentiment, coefficients))
            lp = compute_log_likelihood(feature_matrix, sentiment, coefficients)
            print 'iteration %*d: log likelihood of observed labels = %.8f' % \
                (int(np.ceil(np.log10(max_iter))), itr, lp)
    """
    import matplotlib.pyplot as plt
    x= [i for i in range(len(lplist))]
    plt.plot(x,lplist,'ro')
    plt.show()
    """
    return coefficients
```

### 14. run the logistic regression solver
- feature_matrix = feature_matrix extracted in #9
- sentiment = sentiment extracted in #9
- initial_coefficients = a 194-dimensional vector filled with zeros
- step_size = 1e-7
- max_iter = 301


```python
initial_coefficients = np.zeros((194,1))
step_size = 1e-7
max_iter = 301
```


```python
coefficients = logistic_regression(feature_matrix, sentiment, initial_coefficients, step_size, max_iter)
```

    iteration   0: log likelihood of observed labels = -36780.91768478
    iteration   1: log likelihood of observed labels = -36775.13434712
    iteration   2: log likelihood of observed labels = -36769.35713564
    iteration   3: log likelihood of observed labels = -36763.58603240
    iteration   4: log likelihood of observed labels = -36757.82101962
    iteration   5: log likelihood of observed labels = -36752.06207964
    iteration   6: log likelihood of observed labels = -36746.30919497
    iteration   7: log likelihood of observed labels = -36740.56234821
    iteration   8: log likelihood of observed labels = -36734.82152213
    iteration   9: log likelihood of observed labels = -36729.08669961
    iteration  10: log likelihood of observed labels = -36723.35786366
    iteration  11: log likelihood of observed labels = -36717.63499744
    iteration  12: log likelihood of observed labels = -36711.91808422
    iteration  13: log likelihood of observed labels = -36706.20710739
    iteration  14: log likelihood of observed labels = -36700.50205049
    iteration  15: log likelihood of observed labels = -36694.80289716
    iteration  20: log likelihood of observed labels = -36666.39512033
    iteration  30: log likelihood of observed labels = -36610.01327118
    iteration  40: log likelihood of observed labels = -36554.19728365
    iteration  50: log likelihood of observed labels = -36498.93316099
    iteration  60: log likelihood of observed labels = -36444.20783914
    iteration  70: log likelihood of observed labels = -36390.00909449
    iteration  80: log likelihood of observed labels = -36336.32546144
    iteration  90: log likelihood of observed labels = -36283.14615871
    iteration 100: log likelihood of observed labels = -36230.46102347
    iteration 200: log likelihood of observed labels = -35728.89418769
    iteration 300: log likelihood of observed labels = -35268.51212683
    


```python
coefficients = logistic_regression(feature_matrix, sentiment, initial_coefficients, step_size, max_iter)
```


![png](output_42_0.png)


## 5. Quiz question : 
As each iteration of gradient ascent passes, does the log likelihood increase or decrease?

### Answer
increase

### 15. compute class predictions
- First compute the scores using feature_matrix and coefficients using a dot product.
- Then apply threshold 0 on the scores to compute the class predictions. Refer to the formula above.


```python
"""
feature_matrix: N * D
coefficients: D * 1
predictions: N * 1
"""
predictions = predict_probability(feature_matrix, coefficients)
NumPositive = (predictions > 0.5).sum()
print NumPositive

score = np.dot(feature_matrix, coefficients) # N * 1
print (score > 0).sum()
```

    25126
    25126
    

### 6. Quiz question: 
How many reviews were predicted to have positive sentiment?

### Answer:
25126

## Measuring accuracy


```python
print 0 in products['sentiment']
```

    True
    


```python
print -1 in products['sentiment']
```

    False
    


```python
print np.transpose(predictions.flatten()).shape
print (products['sentiment']).shape
```

    (53072L,)
    (53072L,)
    


```python
print (np.transpose(predictions.flatten()))[:5]
```

    [ 0.51275866  0.49265935  0.50602867  0.50196725  0.53290719]
    


```python
correct_num = np.sum((np.transpose(predictions.flatten())> 0.5) == np.array(products['sentiment']>0))
total_num = len(products['sentiment'])
print "correct_num: {}, total_num: {}".format(correct_num, total_num)
accuracy = correct_num * 1./ total_num
print accuracy
```

    correct_num: 39903, total_num: 53072
    0.751865390413
    


```python
np.transpose(predictions.flatten())> 0.5

```




    array([ True, False,  True, ..., False,  True, False], dtype=bool)




```python
np.array(products['sentiment']>0)
```




    array([ True,  True,  True, ..., False, False, False], dtype=bool)




```python
correct_num = np.sum((np.transpose(score.flatten())> 0) == np.array(products['sentiment']>0))
total_num = len(products['sentiment'])
print "correct_num: {}, total_num: {}".format(correct_num, total_num)
accuracy = correct_num * 1./ total_num
print accuracy
```

    correct_num: 39903, total_num: 53072
    0.751865390413
    

### 7. Quiz question: 
What is the accuracy of the model on predictions made above? (round to 2 digits of accuracy)

### Answer:
0.75

## Which words contribute most to positive & negative sentiments

### 17.compute the "most positive words"
- Treat each coefficient as a tuple, i.e. (word, coefficient_value). The intercept has no corresponding word, so throw it out.
- Sort all the (word, coefficient_value) tuples by coefficient_value in descending order. Save the sorted list of tuples to word_coefficient_tuples.


```python
coefficients = list(coefficients[1:]) # exclude intercept
word_coefficient_tuples = [(word, coefficient) for word, coefficient in zip(important_words, coefficients)]
word_coefficient_tuples = sorted(word_coefficient_tuples, key=lambda x:x[1], reverse=True)
```

## Ten "most positive" words

### 18. Compute the 10 most positive words


```python
word_coefficient_tuples[:10]
```




    [(u'great', array([ 0.06654608])),
     (u'love', array([ 0.06589076])),
     (u'easy', array([ 0.06479459])),
     (u'little', array([ 0.04543563])),
     (u'loves', array([ 0.0449764])),
     (u'well', array([ 0.030135])),
     (u'perfect', array([ 0.02973994])),
     (u'old', array([ 0.02007754])),
     (u'nice', array([ 0.01840871])),
     (u'daughter', array([ 0.0177032]))]



### 8. Quiz question: 
Which word is not present in the top 10 "most positive" words?

## Ten "most negative" words


```python
word_coefficient_tuples[-10:]
```




    [(u'monitor', array([-0.0244821])),
     (u'return', array([-0.02659278])),
     (u'back', array([-0.0277427])),
     (u'get', array([-0.02871155])),
     (u'disappointed', array([-0.02897898])),
     (u'even', array([-0.03005125])),
     (u'work', array([-0.03306952])),
     (u'money', array([-0.03898204])),
     (u'product', array([-0.04151103])),
     (u'would', array([-0.05386015]))]



### 9. Quiz question: 
Which word is not present in the top 10 "most negative" words?


```python
print np.array([1,2,3])==np.array([1,3,2])
```

    [ True False False]
    


```python

```
