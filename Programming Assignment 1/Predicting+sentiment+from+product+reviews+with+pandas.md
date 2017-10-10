

```python
import pandas as pd
import numpy as np
```

## Load Amazon dataset

### 1. 
Load the dataset consisting of baby product reviews on Amazon.com. Store the data in a data frame products.


```python
products = pd.read_csv('amazon_baby.csv')
```

## Perform text cleaning

### 2. 
We start by removing punctuation, so that words "cake." and "cake!" are counted as the same word.


```python
products = products.fillna({'review':''})  # fill in N/A's in the review column
```


```python
def remove_punctuation(text):
    import string
    return text.translate(None, string.punctuation) 

products['review_clean'] = products['review'].apply(remove_punctuation)
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Planetwise Flannel Wipes</td>
      <td>These flannel wipes are OK, but in my opinion ...</td>
      <td>3</td>
      <td>These flannel wipes are OK but in my opinion n...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Planetwise Wipe Pouch</td>
      <td>it came early and was not disappointed. i love...</td>
      <td>5</td>
      <td>it came early and was not disappointed i love ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Annas Dream Full Quilt with 2 Shams</td>
      <td>Very soft and comfortable and warmer than it l...</td>
      <td>5</td>
      <td>Very soft and comfortable and warmer than it l...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Stop Pacifier Sucking without tears with Thumb...</td>
      <td>This is a product well worth the purchase.  I ...</td>
      <td>5</td>
      <td>This is a product well worth the purchase  I h...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Stop Pacifier Sucking without tears with Thumb...</td>
      <td>All of my kids have cried non-stop when I trie...</td>
      <td>5</td>
      <td>All of my kids have cried nonstop when I tried...</td>
    </tr>
  </tbody>
</table>
</div>



## Extract Sentiments

### 3. 
We will ignore all reviews with rating = 3, since they tend to have a neutral sentiment. 


```python
products = products[products['rating'] != 3]
```

### 4. 
Now, we will assign reviews with a rating of 4 or higher to be positive reviews, while the ones with rating of 2 or lower are negative. For the sentiment column, we use +1 for the positive class label and -1 for the negative class label. A good way is to create an anonymous function that converts a rating into a class label and then apply that function to every element in the rating column. In SFrame, you would use apply():


```python
products['sentiment'] = products['rating'].apply(lambda rating : +1 if rating > 3 else -1)
```


```python
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
  </tbody>
</table>
</div>



## Split into training and test sets



### 5. 
Let's perform a train/test split with 80% of the data in the training set and 20% of the data in the test set. If you are using SFrame, make sure to use seed=1 so that you get the same result as everyone else does. (This way, you will get the right numbers for the quiz.)


```python
import json
with open('test_data_idx.json') as test_data_file:    
    test_data_idx = json.load(test_data_file)
with open('train_data_idx.json') as train_data_file:    
    train_data_idx = json.load(train_data_file)

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



## Build the word count vector for each review


### 6. 
We will now compute the word count for each word that appears in the reviews. A vector consisting of word counts is often referred to as bag-of-word features. Since most words occur in only a few reviews, word count vectors are sparse. For this reason, scikit-learn and many other tools use sparse matrices to store a collection of word count vectors. Refer to appropriate manuals to produce sparse word count vectors. General steps for extracting word count vectors are as follows:

- Learn a vocabulary (set of all words) from the training data. Only the words that show up in the training data will be considered for feature extraction.
- Compute the occurrences of the words in each review and collect them into a row vector.
- Build a sparse matrix where each row is the word count vector for the corresponding review. Call this matrix train_matrix.
- Using the same mapping between words and columns, convert the test data into a sparse matrix test_matrix.


```python
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
     # Use this token pattern to keep single-letter words
# First, learn vocabulary from the training data and assign columns to words
# Then convert the training data into a sparse matrix
train_matrix = vectorizer.fit_transform(train_data['review_clean'])
# Second, convert the test data into a sparse matrix, using the same word-column mapping
test_matrix = vectorizer.transform(test_data['review_clean'])
#print vectorizer.vocabulary_
```

## Train a sentiment classifier with logistic regression

### 7. 
Learn a logistic regression classifier using the training data. If you are using scikit-learn, you should create an instance of the [LogisticRegression class](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) and then call the method fit() to train the classifier. This model should use the sparse word count matrix (train_matrix) as features and the column sentiment of train_data as the target. Use the default values for other parameters. Call this model sentiment_model.


```python
from sklearn.linear_model import LogisticRegression
sentiment_model = LogisticRegression()
sentiment_model.fit(train_matrix, train_data['sentiment'])
```




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)



### 8. 
There should be over 100,000 coefficients in this sentiment_model. Recall from the lecture that positive weights w_j correspond to weights that cause positive sentiment, while negative weights correspond to negative sentiment. Calculate the number of positive (>= 0, which is actually nonnegative) coefficients.

### Quiz question: 
How many weights are >= 0?


```python
np.sum(sentiment_model.coef_ >= 0)
```




    85928



### Answer 
85928

## Making predictions with logistic regression

### 9. 
Now that a model is trained, we can make predictions on the test data. In this section, we will explore this in the context of 3 data points in the test data. Take the 11th, 12th, and 13th data points in the test data and save them to sample_test_data. The following cell extracts the three data points from the SFrame test_data and print their content:




```python
sample_test_data = test_data.iloc[10:13]
print sample_test_data
```

                                                     name  \
    59                          Our Baby Girl Memory Book   
    71  Wall Decor Removable Decal Sticker - Colorful ...   
    91  New Style Trailing Cherry Blossom Tree Decal R...   
    
                                                   review  rating  \
    59  Absolutely love it and all of the Scripture in...       5   
    71  Would not purchase again or recommend. The dec...       2   
    91  Was so excited to get this product for my baby...       1   
    
                                             review_clean  sentiment  
    59  Absolutely love it and all of the Scripture in...          1  
    71  Would not purchase again or recommend The deca...         -1  
    91  Was so excited to get this product for my baby...         -1  
    


```python
sample_test_data.iloc[0]['review']
```




    'Absolutely love it and all of the Scripture in it.  I purchased the Baby Boy version for my grandson when he was born and my daughter-in-law was thrilled to receive the same book again.'




```python
sample_test_data.iloc[1]['review']
```




    'Would not purchase again or recommend. The decals were thick almost plastic like and were coming off the wall as I was applying them! The would NOT stick! Literally stayed stuck for about 5 minutes then started peeling off.'




```python
sample_test_matrix = vectorizer.transform(sample_test_data['review_clean'])
scores = sentiment_model.decision_function(sample_test_matrix)
print scores

print sentiment_model.predict(sample_test_matrix)
```

    [  5.60988975  -3.13630894 -10.4141069 ]
    [ 1 -1 -1]
    

## Prediciting Sentiment

### 11. 
These scores can be used to make class predictions as follows:

Using scores, write code to calculate predicted labels for sample_test_data.

Checkpoint: Make sure your class predictions match with the ones obtained from sentiment_model. The logistic regression classifier in scikit-learn comes with the predict function for this purpose.

## Probability Predictions


```python
print [1./(1+np.exp(-x)) for x in scores]
```

    [0.99635188446958933, 0.041634146264237344, 3.0005288630612952e-05]
    


```python
print sentiment_model.classes_
print sentiment_model.predict_proba(sample_test_matrix)
```

    [-1  1]
    [[  3.64811553e-03   9.96351884e-01]
     [  9.58365854e-01   4.16341463e-02]
     [  9.99969995e-01   3.00052886e-05]]
    

### Quiz question: 
Of the three data points in sample_test_data, which one (first, second, or third) has the lowest probability of being classified as a positive review?

### Answer:
third

## Find the most positive (and negative) review

### 13. 
We now turn to examining the full test dataset, test_data, and use sklearn.linear_model.LogisticRegression to form predictions on all of the test data points.

Using the sentiment_model, find the 20 reviews in the entire test_data with the highest probability of being classified as a positive review. We refer to these as the "most positive reviews."

To calculate these top-20 reviews, use the following steps:

- Make probability predictions on test_data using the sentiment_model.
- Sort the data according to those predictions and pick the top 20.


## Quiz Question: 
Which of the following products are represented in the 20 most positive reviews?


```python
test_scores = sentiment_model.decision_function(test_matrix)
positive_idx = np.argsort(-test_scores)[:20]
print positive_idx
print test_scores[positive_idx[0]]
test_data.iloc[positive_idx]
```

    [18112 15732 24286 25554 24899  9125 21531 32782 30535  9555 14482 30634
     17558 26830 11923 20743  4140 30076 33060 26838]
    53.8185477823
    




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
      <th>100166</th>
      <td>Infantino Wrap and Tie Baby Carrier, Black Blu...</td>
      <td>I bought this carrier when my daughter was abo...</td>
      <td>5</td>
      <td>I bought this carrier when my daughter was abo...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>87017</th>
      <td>Baby Einstein Around The World Discovery Center</td>
      <td>I am so HAPPY I brought this item for my 7 mon...</td>
      <td>5</td>
      <td>I am so HAPPY I brought this item for my 7 mon...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>133651</th>
      <td>Britax 2012 B-Agile Stroller, Red</td>
      <td>[I got this stroller for my daughter prior to ...</td>
      <td>4</td>
      <td>I got this stroller for my daughter prior to t...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>140816</th>
      <td>Diono RadianRXT Convertible Car Seat, Plum</td>
      <td>I bought this seat for my tall (38in) and thin...</td>
      <td>5</td>
      <td>I bought this seat for my tall 38in and thin 2...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>137034</th>
      <td>Graco Pack 'n Play Element Playard - Flint</td>
      <td>My husband and I assembled this Pack n' Play l...</td>
      <td>4</td>
      <td>My husband and I assembled this Pack n Play la...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>50315</th>
      <td>P'Kolino Silly Soft Seating in Tias, Green</td>
      <td>I've purchased both the P'Kolino Little Reader...</td>
      <td>4</td>
      <td>Ive purchased both the PKolino Little Reader C...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>119182</th>
      <td>Roan Rocco Classic Pram Stroller 2-in-1 with B...</td>
      <td>Great Pram Rocco!!!!!!I bought this pram from ...</td>
      <td>5</td>
      <td>Great Pram RoccoI bought this pram from Europe...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>180646</th>
      <td>Mamas &amp;amp; Papas 2014 Urbo2 Stroller - Black</td>
      <td>After much research I purchased an Urbo2. It's...</td>
      <td>4</td>
      <td>After much research I purchased an Urbo2 Its e...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>168081</th>
      <td>Buttons Cloth Diaper Cover - One Size - 8 Colo...</td>
      <td>We are big Best Bottoms fans here, but I wante...</td>
      <td>4</td>
      <td>We are big Best Bottoms fans here but I wanted...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>52631</th>
      <td>Evenflo X Sport Plus Convenience Stroller - Ch...</td>
      <td>After seeing this in Parent's Magazine and rea...</td>
      <td>5</td>
      <td>After seeing this in Parents Magazine and read...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>80155</th>
      <td>Simple Wishes Hands-Free Breastpump Bra, Pink,...</td>
      <td>I just tried this hands free breastpump bra, a...</td>
      <td>5</td>
      <td>I just tried this hands free breastpump bra an...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>168697</th>
      <td>Graco FastAction Fold Jogger Click Connect Str...</td>
      <td>Graco's FastAction Jogging Stroller definitely...</td>
      <td>5</td>
      <td>Gracos FastAction Jogging Stroller definitely ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>97325</th>
      <td>Freemie Hands-Free Concealable Breast Pump Col...</td>
      <td>I absolutely love this product.  I work as a C...</td>
      <td>5</td>
      <td>I absolutely love this product  I work as a Cu...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>147949</th>
      <td>Baby Jogger City Mini GT Single Stroller, Shad...</td>
      <td>Amazing, Love, Love, Love it !!! All 5 STARS a...</td>
      <td>5</td>
      <td>Amazing Love Love Love it  All 5 STARS all the...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>66059</th>
      <td>Evenflo 6 Pack Classic Glass Bottle, 4-Ounce</td>
      <td>It's always fun to write a review on those pro...</td>
      <td>5</td>
      <td>Its always fun to write a review on those prod...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>114796</th>
      <td>Fisher-Price Cradle 'N Swing,  My Little Snuga...</td>
      <td>My husband and I cannot state enough how much ...</td>
      <td>5</td>
      <td>My husband and I cannot state enough how much ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>22586</th>
      <td>Britax Decathlon Convertible Car Seat, Tiffany</td>
      <td>I researched a few different seats to put in o...</td>
      <td>4</td>
      <td>I researched a few different seats to put in o...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>165593</th>
      <td>Ikea 36 Pcs Kalas Kids Plastic BPA Free Flatwa...</td>
      <td>For the price this set is unbelievable- and tr...</td>
      <td>5</td>
      <td>For the price this set is unbelievable and tru...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>182089</th>
      <td>Summer Infant Wide View Digital Color Video Mo...</td>
      <td>I love this baby monitor.  I can compare this ...</td>
      <td>5</td>
      <td>I love this baby monitor  I can compare this o...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>147996</th>
      <td>Baby Jogger City Mini GT Double Stroller, Shad...</td>
      <td>We are well pleased with this stroller, and I ...</td>
      <td>4</td>
      <td>We are well pleased with this stroller and I w...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



### 14. 
Now, let us repeat this exercise to find the "most negative reviews." Use the prediction probabilities to find the 20 reviews in the test_data with the lowest probability of being classified as a positive review. Repeat the same steps above but make sure you sort in the opposite order.



### Quiz Question: 
Which of the following products are represented in the 20 most negative reviews?


```python
negative_idx = np.argsort(test_scores)[:20]
print negative_idx
print test_scores[negative_idx[0]]
test_data.iloc[negative_idx]
```

    [ 2931 21700 13939  8818 28184 17069  9655 14711 20594  1942  1810 10814
     13751 31226  7310 27231 28120   205 15062  5831]
    -34.6348776854
    




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
      <th>16042</th>
      <td>Fisher-Price Ocean Wonders Aquarium Bouncer</td>
      <td>We have not had ANY luck with Fisher-Price pro...</td>
      <td>2</td>
      <td>We have not had ANY luck with FisherPrice prod...</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>120209</th>
      <td>Levana Safe N'See Digital Video Baby Monitor w...</td>
      <td>This is the first review I have ever written o...</td>
      <td>1</td>
      <td>This is the first review I have ever written o...</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>77072</th>
      <td>Safety 1st Exchangeable Tip 3 in 1 Thermometer</td>
      <td>I thought it sounded great to have different t...</td>
      <td>1</td>
      <td>I thought it sounded great to have different t...</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>48694</th>
      <td>Adiri BPA Free Natural Nurser Ultimate Bottle ...</td>
      <td>I will try to write an objective review of the...</td>
      <td>2</td>
      <td>I will try to write an objective review of the...</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>155287</th>
      <td>VTech Communications Safe &amp;amp; Sounds Full Co...</td>
      <td>This is my second video monitoring system, the...</td>
      <td>1</td>
      <td>This is my second video monitoring system the ...</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>94560</th>
      <td>The First Years True Choice P400 Premium Digit...</td>
      <td>Note: we never installed batteries in these un...</td>
      <td>1</td>
      <td>Note we never installed batteries in these uni...</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>53207</th>
      <td>Safety 1st High-Def Digital Monitor</td>
      <td>We bought this baby monitor to replace a diffe...</td>
      <td>1</td>
      <td>We bought this baby monitor to replace a diffe...</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>81332</th>
      <td>Cloth Diaper Sprayer--styles may vary</td>
      <td>I bought this sprayer out of desperation durin...</td>
      <td>1</td>
      <td>I bought this sprayer out of desperation durin...</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>113995</th>
      <td>Motorola Digital Video Baby Monitor with Room ...</td>
      <td>DO NOT BUY THIS BABY MONITOR!I purchased this ...</td>
      <td>1</td>
      <td>DO NOT BUY THIS BABY MONITORI purchased this m...</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>10677</th>
      <td>Philips AVENT Newborn Starter Set</td>
      <td>It's 3am in the morning and needless to say, t...</td>
      <td>1</td>
      <td>Its 3am in the morning and needless to say thi...</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>9915</th>
      <td>Cosco Alpha Omega Elite Convertible Car Seat</td>
      <td>I bought this car seat after both seeing  the ...</td>
      <td>1</td>
      <td>I bought this car seat after both seeing  the ...</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>59546</th>
      <td>Ellaroo Mei Tai Baby Carrier - Hershey</td>
      <td>This is basically an overpriced piece of fabri...</td>
      <td>1</td>
      <td>This is basically an overpriced piece of fabri...</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>75994</th>
      <td>Peg-Perego Tatamia High Chair, White Latte</td>
      <td>I can see why there are so many good reviews o...</td>
      <td>2</td>
      <td>I can see why there are so many good reviews o...</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>172090</th>
      <td>Belkin WeMo Wi-Fi Baby Monitor for Apple iPhon...</td>
      <td>I read so many reviews saying the Belkin WiFi ...</td>
      <td>2</td>
      <td>I read so many reviews saying the Belkin WiFi ...</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>40079</th>
      <td>Chicco Cortina KeyFit 30 Travel System in Adve...</td>
      <td>My wife and I have used this system in two car...</td>
      <td>1</td>
      <td>My wife and I have used this system in two car...</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>149987</th>
      <td>NUK Cook-n-Blend Baby Food Maker</td>
      <td>It thought this would be great. I did a lot of...</td>
      <td>1</td>
      <td>It thought this would be great I did a lot of ...</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>154878</th>
      <td>VTech Communications Safe &amp;amp; Sound Digital ...</td>
      <td>First, the distance on these are no more than ...</td>
      <td>1</td>
      <td>First the distance on these are no more than 7...</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>1116</th>
      <td>Safety 1st Deluxe 4-in-1 Bath Station</td>
      <td>This item is junk.  I originally chose it beca...</td>
      <td>1</td>
      <td>This item is junk  I originally chose it becau...</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>83234</th>
      <td>Thirsties Hemp Inserts 2 Pack, Small 6-18 Lbs</td>
      <td>My Experience: Babykicks Inserts failure vs RA...</td>
      <td>5</td>
      <td>My Experience Babykicks Inserts failure vs RAV...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>31741</th>
      <td>Regalo My Cot Portable Bed, Royal Blue</td>
      <td>If I could give this product zero stars I woul...</td>
      <td>1</td>
      <td>If I could give this product zero stars I woul...</td>
      <td>-1</td>
    </tr>
  </tbody>
</table>
</div>



## Compute accuracy of the classifier

### 15. 
We will now evaluate the accuracy of the trained classifier. Recall that the accuracy is given by

$$ accuracy=\frac{\# correctly classified examples}{ \# total examples}$$
This can be computed as follows:

- Step 1: Use the sentiment_model to compute class predictions.
- Step 2: Count the number of data points when the predicted class labels match the ground truth labels.
- Step 3: Divide the total number of correct predictions by the total number of data points in the dataset.


### Quiz Question: 
What is the accuracy of the sentiment_model on the test_data? Round your answer to 2 decimal places (e.g. 0.76).

### Quiz Question: 
Does a higher accuracy value on the training_data always imply that the classifier is better?


```python
predicted_y = sentiment_model.predict(test_matrix)
correct_num = np.sum(predicted_y == test_data['sentiment'])
total_num = len(test_data['sentiment'])
print "correct_num: {}, total_num: {}".format(correct_num, total_num)
accuracy = correct_num * 1./ total_num
print accuracy
```

    correct_num: 31077, total_num: 33336
    0.932235421166
    

### Answer:
0.93

## Learn another classifier with fewer words

### 16. 
There were a lot of words in the model we trained above. We will now train a simpler logistic regression model using only a subet of words that occur in the reviews. For this assignment, we selected 20 words to work with. 


```python
significant_words = ['love', 'great', 'easy', 'old', 'little', 'perfect', 'loves', 
      'well', 'able', 'car', 'broke', 'less', 'even', 'waste', 'disappointed', 
      'work', 'product', 'money', 'would', 'return']
```


```python
vectorizer_word_subset = CountVectorizer(vocabulary=significant_words) # limit to 20 words
train_matrix_word_subset = vectorizer_word_subset.fit_transform(train_data['review_clean'])
test_matrix_word_subset = vectorizer_word_subset.transform(test_data['review_clean'])
```

## Train a logistic regression model on a subset of data

### 17. 
Now build a logistic regression classifier with train_matrix_word_subset as features and sentiment as the target. Call this model simple_model.




```python
simple_model = LogisticRegression()
simple_model.fit(train_matrix_word_subset, train_data['sentiment'])
```




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)



### 18. 
Let us inspect the weights (coefficients) of the simple_model. First, build a table to store (word, coefficient) pairs. If you are using SFrame with scikit-learn, you can combine words with coefficients by running


```python
simple_model_coef_table = pd.DataFrame({'word':significant_words,
                                         'coefficient':simple_model.coef_.flatten()})
#simple_model_coef_table
simple_model_coef_table.sort_values(['coefficient'], ascending=False)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>coefficient</th>
      <th>word</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6</th>
      <td>1.673074</td>
      <td>loves</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.509812</td>
      <td>perfect</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1.363690</td>
      <td>love</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.192538</td>
      <td>easy</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.944000</td>
      <td>great</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.520186</td>
      <td>little</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.503760</td>
      <td>well</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.190909</td>
      <td>able</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.085513</td>
      <td>old</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.058855</td>
      <td>car</td>
    </tr>
    <tr>
      <th>11</th>
      <td>-0.209563</td>
      <td>less</td>
    </tr>
    <tr>
      <th>16</th>
      <td>-0.320556</td>
      <td>product</td>
    </tr>
    <tr>
      <th>18</th>
      <td>-0.362167</td>
      <td>would</td>
    </tr>
    <tr>
      <th>12</th>
      <td>-0.511380</td>
      <td>even</td>
    </tr>
    <tr>
      <th>15</th>
      <td>-0.621169</td>
      <td>work</td>
    </tr>
    <tr>
      <th>17</th>
      <td>-0.898031</td>
      <td>money</td>
    </tr>
    <tr>
      <th>10</th>
      <td>-1.651576</td>
      <td>broke</td>
    </tr>
    <tr>
      <th>13</th>
      <td>-2.033699</td>
      <td>waste</td>
    </tr>
    <tr>
      <th>19</th>
      <td>-2.109331</td>
      <td>return</td>
    </tr>
    <tr>
      <th>14</th>
      <td>-2.348298</td>
      <td>disappointed</td>
    </tr>
  </tbody>
</table>
</div>



### Quiz Question: 
Consider the coefficients of simple_model. How many of the 20 coefficients (corresponding to the 20 significant_words) are positive for the simple_model?


```python
len(simple_model_coef_table[simple_model_coef_table['coefficient']>0])
```




    10



### Answer: 
10

### Quiz Question: 
Are the positive words in the simple_model also positive words in the sentiment_model?


```python
model_coef_table = pd.DataFrame({'word':significant_words,
                                         'coefficient':simple_model.coef_.flatten()})
#simple_model_coef_table
simple_model_coef_table.sort_values(['coefficient'], ascending=False)
```


```python
vectorizer_word_subset.get_feature_names()
```




    ['love',
     'great',
     'easy',
     'old',
     'little',
     'perfect',
     'loves',
     'well',
     'able',
     'car',
     'broke',
     'less',
     'even',
     'waste',
     'disappointed',
     'work',
     'product',
     'money',
     'would',
     'return']



### Answer:

## Comparing models

### 19. 
We will now compare the accuracy of the sentiment_model and the simple_model.

First, compute the classification accuracy of the sentiment_model on the train_data.

Now, compute the classification accuracy of the simple_model on the train_data.




```python
train_predicted_y = sentiment_model.predict(train_matrix)
correct_num = np.sum(train_predicted_y == train_data['sentiment'])
total_num = len(train_data['sentiment'])
print "correct_num: {}, total_num: {}".format(correct_num, total_num)
train_accuracy = correct_num * 1./ total_num
print "sentiment_model training accuracy: {}".format(train_accuracy)

train_predicted_y = simple_model.predict(train_matrix_word_subset)
correct_num = np.sum(train_predicted_y == train_data['sentiment'])
total_num = len(train_data['sentiment'])
print "correct_num: {}, total_num: {}".format(correct_num, total_num)
train_accuracy = correct_num * 1./ total_num
print "simple_model training accuracy: {}".format(train_accuracy)
```

    correct_num: 129159, total_num: 133416
    sentiment_model training accuracy: 0.968092282785
    correct_num: 115648, total_num: 133416
    simple_model training accuracy: 0.866822570007
    

### Quiz Question:
Which model (sentiment_model or simple_model) has higher accuracy on the TRAINING set?

### Answer:
sentiment_model 

### 20. 
Now, we will repeat this exercise on the test_data. Start by computing the classification accuracy of the sentiment_model on the test_data.

Next, compute the classification accuracy of the simple_model on the test_data.




```python
test_predicted_y = sentiment_model.predict(test_matrix)
correct_num = np.sum(test_predicted_y == test_data['sentiment'])
total_num = len(test_data['sentiment'])
print "correct_num: {}, total_num: {}".format(correct_num, total_num)
test_accuracy = correct_num * 1./ total_num
print "sentiment_model test accuracy: {}".format(test_accuracy)

test_predicted_y = simple_model.predict(test_matrix_word_subset)
correct_num = np.sum(test_predicted_y == test_data['sentiment'])
total_num = len(test_data['sentiment'])
print "correct_num: {}, total_num: {}".format(correct_num, total_num)
test_accuracy = correct_num * 1./ total_num
print "simple_model test accuracy: {}".format(test_accuracy)
```

    correct_num: 31077, total_num: 33336
    sentiment_model test accuracy: 0.932235421166
    correct_num: 28981, total_num: 33336
    simple_model test accuracy: 0.869360451164
    

### Quiz Question: 
Which model (sentiment_model or simple_model) has higher accuracy on the TEST set?

### Answer:
sentiment_model 

## Baseline: Majority class prediction

### 21. 
It is quite common to use the majority class classifier as the a baseline (or reference) model for comparison with your classifier model. The majority classifier model predicts the majority class for all data points. At the very least, you should healthily beat the majority class classifier, otherwise, the model is (usually) pointless.



```python
positive_label = len(test_data[test_data['sentiment']>0])
negative_label = len(test_data[test_data['sentiment']<0])
print "positive_label is {}, negative_label is {}".format(positive_label, negative_label)
```

    positive_label is 28095, negative_label is 5241
    


```python
baseline_accuracy = positive_label*1./(positive_label+negative_label)
print "baseline_accuracy is {}".format(baseline_accuracy)
```

    baseline_accuracy is 0.842782577394
    


### Quiz Question:
Enter the accuracy of the majority class classifier model on the test_data. Round your answer to two decimal places (e.g. 0.76).



### Answer:
0.84

### Quiz Question:
Is the sentiment_model definitely better than the majority class classifier (the baseline)?

### Answer:
Yes
