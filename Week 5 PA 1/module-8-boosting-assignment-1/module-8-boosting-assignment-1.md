
# Exploring Ensemble Methods

In this assignment, we will explore the use of boosting. We will use the pre-implemented gradient boosted trees in GraphLab Create. You will:

* Use SFrames to do some feature engineering.
* Train a boosted ensemble of decision-trees (gradient boosted trees) on the LendingClub dataset.
* Predict whether a loan will default along with prediction probabilities (on a validation set).
* Evaluate the trained model and compare it with a baseline.
* Find the most positive and negative loans using the learned model.
* Explore how the number of trees influences classification performance.

Let's get started!

## Fire up Graphlab Create


```python
import numpy as np
import pandas as pd
import json
```

# Load LendingClub dataset

We will be using the [LendingClub](https://www.lendingclub.com/) data. As discussed earlier, the [LendingClub](https://www.lendingclub.com/) is a peer-to-peer leading company that directly connects borrowers and potential lenders/investors. 

Just like we did in previous assignments, we will build a classification model to predict whether or not a loan provided by lending club is likely to default.

Let us start by loading the data.


```python
loans = pd.read_csv('lending-club-data.csv')
loans.head(2)
```

    C:\Users\SSQ\AppData\Roaming\Python\Python27\site-packages\IPython\core\interactiveshell.py:2717: DtypeWarning: Columns (19,47) have mixed types. Specify dtype option on import or set low_memory=False.
      interactivity=interactivity, compiler=compiler, result=result)
    




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>member_id</th>
      <th>loan_amnt</th>
      <th>funded_amnt</th>
      <th>funded_amnt_inv</th>
      <th>term</th>
      <th>int_rate</th>
      <th>installment</th>
      <th>grade</th>
      <th>sub_grade</th>
      <th>...</th>
      <th>sub_grade_num</th>
      <th>delinq_2yrs_zero</th>
      <th>pub_rec_zero</th>
      <th>collections_12_mths_zero</th>
      <th>short_emp</th>
      <th>payment_inc_ratio</th>
      <th>final_d</th>
      <th>last_delinq_none</th>
      <th>last_record_none</th>
      <th>last_major_derog_none</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1077501</td>
      <td>1296599</td>
      <td>5000</td>
      <td>5000</td>
      <td>4975</td>
      <td>36 months</td>
      <td>10.65</td>
      <td>162.87</td>
      <td>B</td>
      <td>B2</td>
      <td>...</td>
      <td>0.4</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0</td>
      <td>8.1435</td>
      <td>20141201T000000</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1077430</td>
      <td>1314167</td>
      <td>2500</td>
      <td>2500</td>
      <td>2500</td>
      <td>60 months</td>
      <td>15.27</td>
      <td>59.83</td>
      <td>C</td>
      <td>C4</td>
      <td>...</td>
      <td>0.8</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1</td>
      <td>2.3932</td>
      <td>20161201T000000</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 68 columns</p>
</div>



Let's quickly explore what the dataset looks like. First, let's print out the column names to see what features we have in this dataset. We have done this in previous assignments, so we won't belabor this here.


```python
loans.columns
```




    Index([u'id', u'member_id', u'loan_amnt', u'funded_amnt', u'funded_amnt_inv',
           u'term', u'int_rate', u'installment', u'grade', u'sub_grade',
           u'emp_title', u'emp_length', u'home_ownership', u'annual_inc',
           u'is_inc_v', u'issue_d', u'loan_status', u'pymnt_plan', u'url', u'desc',
           u'purpose', u'title', u'zip_code', u'addr_state', u'dti',
           u'delinq_2yrs', u'earliest_cr_line', u'inq_last_6mths',
           u'mths_since_last_delinq', u'mths_since_last_record', u'open_acc',
           u'pub_rec', u'revol_bal', u'revol_util', u'total_acc',
           u'initial_list_status', u'out_prncp', u'out_prncp_inv', u'total_pymnt',
           u'total_pymnt_inv', u'total_rec_prncp', u'total_rec_int',
           u'total_rec_late_fee', u'recoveries', u'collection_recovery_fee',
           u'last_pymnt_d', u'last_pymnt_amnt', u'next_pymnt_d',
           u'last_credit_pull_d', u'collections_12_mths_ex_med',
           u'mths_since_last_major_derog', u'policy_code', u'not_compliant',
           u'status', u'inactive_loans', u'bad_loans', u'emp_length_num',
           u'grade_num', u'sub_grade_num', u'delinq_2yrs_zero', u'pub_rec_zero',
           u'collections_12_mths_zero', u'short_emp', u'payment_inc_ratio',
           u'final_d', u'last_delinq_none', u'last_record_none',
           u'last_major_derog_none'],
          dtype='object')



## Modifying the target column

The target column (label column) of the dataset that we are interested in is called `bad_loans`. In this column **1** means a risky (bad) loan **0** means a safe  loan.

As in past assignments, in order to make this more intuitive and consistent with the lectures, we reassign the target to be:
* **+1** as a safe  loan, 
* **-1** as a risky (bad) loan. 

We put this in a new column called `safe_loans`.


```python
loans['safe_loans'] = loans['bad_loans'].apply(lambda x : +1 if x==0 else -1)
loans = loans.drop('bad_loans', axis=1)
```

## Selecting features

In this assignment, we will be using a subset of features (categorical and numeric). The features we will be using are **described in the code comments** below. If you are a finance geek, the [LendingClub](https://www.lendingclub.com/) website has a lot more details about these features.

The features we will be using are described in the code comments below:


```python
target = 'safe_loans'
features = ['grade',                     # grade of the loan (categorical)
            'sub_grade_num',             # sub-grade of the loan as a number from 0 to 1
            'short_emp',                 # one year or less of employment
            'emp_length_num',            # number of years of employment
            'home_ownership',            # home_ownership status: own, mortgage or rent
            'dti',                       # debt to income ratio
            'purpose',                   # the purpose of the loan
            'payment_inc_ratio',         # ratio of the monthly payment to income
            'delinq_2yrs',               # number of delinquincies 
            'delinq_2yrs_zero',          # no delinquincies in last 2 years
            'inq_last_6mths',            # number of creditor inquiries in last 6 months
            'last_delinq_none',          # has borrower had a delinquincy
            'last_major_derog_none',     # has borrower had 90 day or worse rating
            'open_acc',                  # number of open credit accounts
            'pub_rec',                   # number of derogatory public records
            'pub_rec_zero',              # no derogatory public records
            'revol_util',                # percent of available credit being used
            'total_rec_late_fee',        # total late fees received to day
            'int_rate',                  # interest rate of the loan
            'total_rec_int',             # interest received to date
            'annual_inc',                # annual income of borrower
            'funded_amnt',               # amount committed to the loan
            'funded_amnt_inv',           # amount committed by investors for the loan
            'installment',               # monthly payment owed by the borrower
           ]
```

## Skipping observations with missing values

Recall from the lectures that one common approach to coping with missing values is to **skip** observations that contain missing values.

We run the following code to do so:


```python
print loans.shape
loans = loans[[target] + features].dropna()
print loans.shape
```

    (122607, 68)
    (122578, 25)
    

Fortunately, there are not too many missing values. We are retaining most of the data.

## Transform categorical data into binary features

Since we are implementing binary decision trees, we transform our categorical data into binary data using 1-hot encoding, just as in the previous assignment. Here is the summary of that discussion:

For instance, the home_ownership feature represents the home ownership status of the loanee, which is either own, mortgage or rent. For example, if a data point has the feature
  
  {'home_ownership': 'RENT'}

we want to turn this into three features:
 
 { 
   'home_ownership = OWN'      : 0, 
   'home_ownership = MORTGAGE' : 0, 
   'home_ownership = RENT'     : 1
 }
 
Since this code requires a few Python and GraphLab tricks, feel free to use this block of code as is. Refer to the API documentation for a deeper understanding.


```python
categorical_variables = []
for feat_name, feat_type in zip(loans.columns, loans.dtypes):
    if feat_type == object:
        categorical_variables.append(feat_name)
        
for feature in categorical_variables:
    
    loans_one_hot_encoded = pd.get_dummies(loans[feature],prefix=feature)
    loans_one_hot_encoded.fillna(0)
    #print loans_one_hot_encoded
    
    loans = loans.drop(feature, axis=1)
    for col in loans_one_hot_encoded.columns:
        loans[col] = loans_one_hot_encoded[col]
    
print loans.head(2)        
print loans.columns
```

       safe_loans  sub_grade_num  short_emp  emp_length_num    dti  \
    0           1            0.4          0              11  27.65   
    1          -1            0.8          1               1   1.00   
    
       payment_inc_ratio  delinq_2yrs  delinq_2yrs_zero  inq_last_6mths  \
    0             8.1435          0.0               1.0             1.0   
    1             2.3932          0.0               1.0             5.0   
    
       last_delinq_none       ...         purpose_debt_consolidation  \
    0                 1       ...                                  0   
    1                 1       ...                                  0   
    
       purpose_home_improvement  purpose_house  purpose_major_purchase  \
    0                         0              0                       0   
    1                         0              0                       0   
    
       purpose_medical  purpose_moving  purpose_other  purpose_small_business  \
    0                0               0              0                       0   
    1                0               0              0                       0   
    
       purpose_vacation  purpose_wedding  
    0                 0                0  
    1                 0                0  
    
    [2 rows x 45 columns]
    Index([u'safe_loans', u'sub_grade_num', u'short_emp', u'emp_length_num',
           u'dti', u'payment_inc_ratio', u'delinq_2yrs', u'delinq_2yrs_zero',
           u'inq_last_6mths', u'last_delinq_none', u'last_major_derog_none',
           u'open_acc', u'pub_rec', u'pub_rec_zero', u'revol_util',
           u'total_rec_late_fee', u'int_rate', u'total_rec_int', u'annual_inc',
           u'funded_amnt', u'funded_amnt_inv', u'installment', u'grade_A',
           u'grade_B', u'grade_C', u'grade_D', u'grade_E', u'grade_F', u'grade_G',
           u'home_ownership_MORTGAGE', u'home_ownership_OTHER',
           u'home_ownership_OWN', u'home_ownership_RENT', u'purpose_car',
           u'purpose_credit_card', u'purpose_debt_consolidation',
           u'purpose_home_improvement', u'purpose_house',
           u'purpose_major_purchase', u'purpose_medical', u'purpose_moving',
           u'purpose_other', u'purpose_small_business', u'purpose_vacation',
           u'purpose_wedding'],
          dtype='object')
    

## Split data into training and validation sets

We split the data into training data and validation data. We used `seed=1` to make sure everyone gets the same results. We will use the validation data to help us select model parameters.


```python
with open('module-8-assignment-1-train-idx.json') as train_data_file:    
    train_idx  = json.load(train_data_file)
with open('module-8-assignment-1-validation-idx.json') as validation_data_file:    
    validation_idx = json.load(validation_data_file)

print train_idx[:3]
print validation_idx[:3]
```

    [1, 6, 7]
    [24, 41, 60]
    


```python
print len(train_idx)
print len(validation_idx)
```

    37219
    9284
    


```python
train_data = loans.iloc[train_idx]
validation_data = loans.iloc[validation_idx]
```


```python
print len(loans.dtypes )
```

    45
    

# Gradient boosted tree classifier

Gradient boosted trees are a powerful variant of boosting methods; they have been used to win many [Kaggle](https://www.kaggle.com/) competitions, and have been widely used in industry.  We will explore the predictive power of multiple decision trees as opposed to a single decision tree.

**Additional reading:** If you are interested in gradient boosted trees, here is some additional reading material:
* [GraphLab Create user guide](https://dato.com/learn/userguide/supervised-learning/boosted_trees_classifier.html)
* [Advanced material on boosted trees](http://homes.cs.washington.edu/~tqchen/pdf/BoostedTree.pdf)


We will now train models to predict `safe_loans` using the features above. In this section, we will experiment with training an ensemble of 5 trees. To cap the ensemble classifier at 5 trees, we call the function with **max_iterations=5** (recall that each iterations corresponds to adding a tree). We set `validation_set=None` to make sure everyone gets the same results.

Now, let's use the built-in scikit learn gradient boosting classifier ([sklearn.ensemble.GradientBoostingClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)) to create a gradient boosted classifier on the training data. You will need to import sklearn, sklearn.ensemble, and numpy.

You will have to first convert the SFrame into a numpy data matrix. See the [API](https://turi.com/products/create/docs/generated/graphlab.SFrame.to_numpy.html) for more information. You will also have to extract the label column. Make sure to set max_depth=6 and n_estimators=5.


```python
import sklearn
from sklearn.ensemble import GradientBoostingClassifier
train_Y = train_data['safe_loans'].as_matrix()
train_X = train_data.drop('safe_loans', axis=1).as_matrix()
print train_Y.shape
print train_X.shape

model_5 = GradientBoostingClassifier(n_estimators=5, max_depth=6).fit(train_X, train_Y)

```

    (37219L,)
    (37219L, 44L)
    

# Making predictions

Just like we did in previous sections, let us consider a few positive and negative examples **from the validation set**. We will do the following:
* Predict whether or not a loan is likely to default.
* Predict the probability with which the loan is likely to default.


```python
# Select all positive and negative examples.
validation_safe_loans = validation_data[validation_data[target] == 1]
validation_risky_loans = validation_data[validation_data[target] == -1]

# Select 2 examples from the validation set for positive & negative loans
sample_validation_data_risky = validation_risky_loans[0:2]
sample_validation_data_safe = validation_safe_loans[0:2]

# Append the 4 examples into a single dataset
sample_validation_data = sample_validation_data_safe.append(sample_validation_data_risky)
sample_validation_data
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>safe_loans</th>
      <th>sub_grade_num</th>
      <th>short_emp</th>
      <th>emp_length_num</th>
      <th>dti</th>
      <th>payment_inc_ratio</th>
      <th>delinq_2yrs</th>
      <th>delinq_2yrs_zero</th>
      <th>inq_last_6mths</th>
      <th>last_delinq_none</th>
      <th>...</th>
      <th>purpose_debt_consolidation</th>
      <th>purpose_home_improvement</th>
      <th>purpose_house</th>
      <th>purpose_major_purchase</th>
      <th>purpose_medical</th>
      <th>purpose_moving</th>
      <th>purpose_other</th>
      <th>purpose_small_business</th>
      <th>purpose_vacation</th>
      <th>purpose_wedding</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>22</th>
      <td>1</td>
      <td>0.2</td>
      <td>0</td>
      <td>3</td>
      <td>29.44</td>
      <td>6.30496</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1</td>
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
    <tr>
      <th>26</th>
      <td>1</td>
      <td>0.6</td>
      <td>1</td>
      <td>1</td>
      <td>12.19</td>
      <td>13.49520</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1</td>
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
    <tr>
      <th>24</th>
      <td>-1</td>
      <td>0.4</td>
      <td>0</td>
      <td>3</td>
      <td>13.97</td>
      <td>2.96736</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>41</th>
      <td>-1</td>
      <td>1.0</td>
      <td>0</td>
      <td>11</td>
      <td>16.33</td>
      <td>1.90524</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
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
<p>4 rows × 45 columns</p>
</div>



### Predicting on sample validation data

For each row in the **sample_validation_data**, write code to make **model_5** predict whether or not the loan is classified as a **safe loan**.

**Hint:** Use the `predict` method in `model_5` for this.

For each row in the sample_validation_data, write code to make model_5 predict whether or not the loan is classified as a safe loan. (Hint: if you are using scikit-learn, you can use the [.predict()](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier.predict) method)


```python
model_5.predict(sample_validation_data.drop('safe_loans', axis=1).as_matrix())
```




    array([ 1,  1, -1,  1], dtype=int64)



**Quiz Question:** What percentage of the predictions on `sample_validation_data` did `model_5` get correct?

**0.75**

### Prediction probabilities

For each row in the **sample_validation_data**, what is the probability (according **model_5**) of a loan being classified as **safe**? 

**Hint:** Set `output_type='probability'` to make **probability** predictions using `model_5` on `sample_validation_data`:

For each row in the sample_validation_data, what is the probability (according model_5) of a loan being classified as safe? (Hint: if you are using scikit-learn, you can use the [.predict_proba()](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier.predict_proba) method)


```python
model_5.predict_proba(sample_validation_data.drop('safe_loans', axis=1).as_matrix())
```




    array([[ 0.41642331,  0.58357669],
           [ 0.46949689,  0.53050311],
           [ 0.53807792,  0.46192208],
           [ 0.39591639,  0.60408361]])



**Quiz Question:** According to **model_5**, which loan is the least likely to be a safe loan?

**3**

**Checkpoint:** Can you verify that for all the predictions with `probability >= 0.5`, the model predicted the label **+1**?

## Evaluating the model on the validation data

Recall that the accuracy is defined as follows:
$$
\mbox{accuracy} = \frac{\mbox{# correctly classified examples}}{\mbox{# total examples}}
$$

Evaluate the accuracy of the **model_5** on the **validation_data**.

**Hint**: Use the `.evaluate()` method in the model.

Evaluate the accuracy of the model_5 on the validation_data. (Hint: if you are using scikit-learn, you can use the [.score()](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier.score) method)


```python
validation_Y = validation_data['safe_loans'].as_matrix()
validation_X = validation_data.drop('safe_loans', axis=1).as_matrix()
print validation_X.shape
print validation_Y.shape
```

    (9284L, 44L)
    (9284L,)
    


```python
model_5.score(validation_X,validation_Y)
```




    0.66135286514433433



Calculate the number of **false positives** made by the model.


```python
predictions = model_5.predict(validation_X)
print type(predictions)
print type(validation_Y)
```

    <type 'numpy.ndarray'>
    <type 'numpy.ndarray'>
    


```python
false_positives = ((predictions==1) * (validation_Y==-1)).sum()
print false_positives
```

    1653
    

**Quiz Question**: What is the number of **false positives** on the **validation_data**?

Calculate the number of **false negatives** made by the model.


```python
false_negatives = ((predictions==-1) * (validation_Y==1)).sum()
print false_negatives
```

    1491
    

## Comparison with decision trees

In the earlier assignment, we saw that the prediction accuracy of the decision trees was around **0.64** (rounded). In this assignment, we saw that **model_5** has an accuracy of **0.67** (rounded).

Here, we quantify the benefit of the extra 3% increase in accuracy of **model_5** in comparison with a single decision tree from the original decision tree assignment.

As we explored in the earlier assignment, we calculated the cost of the mistakes made by the model. We again consider the same costs as follows:

* **False negatives**: Assume a cost of \$10,000 per false negative.
* **False positives**: Assume a cost of \$20,000 per false positive.

Assume that the number of false positives and false negatives for the learned decision tree was

* **False negatives**: 1936
* **False positives**: 1503

Using the costs defined above and the number of false positives and false negatives for the decision tree, we can calculate the total cost of the mistakes made by the decision tree model as follows:

```
cost = $10,000 * 1936  + $20,000 * 1503 = $49,420,000
```

The total cost of the mistakes of the model is $49.42M. That is a **lot of money**!.

**Quiz Question**: Using the same costs of the false positives and false negatives, what is the cost of the mistakes made by the boosted tree model (**model_5**) as evaluated on the **validation_set**?


```python
print 10000 * false_negatives + 20000 * false_positives
```

    47970000
    

**Reminder**: Compare the cost of the mistakes made by the boosted trees model with the decision tree model. The extra 3% improvement in prediction accuracy can translate to several million dollars!  And, it was so easy to get by simply boosting our decision trees.

## Most positive & negative loans.

In this section, we will find the loans that are most likely to be predicted **safe**. We can do this in a few steps:

* **Step 1**: Use the **model_5** (the model with 5 trees) and make **probability predictions** for all the loans in the **validation_data**.
* **Step 2**: Similar to what we did in the very first assignment, add the probability predictions as a column called **predictions** into the validation_data.
* **Step 3**: Sort the data (in descreasing order) by the probability predictions.

Start here with **Step 1** & **Step 2**. Make predictions using **model_5** for examples in the **validation_data**. Use `output_type = probability`.

Now, we are ready to go to **Step 3**. You can now use the `prediction` column to sort the loans in **validation_data** (in descending order) by prediction probability. Find the top 5 loans with the highest probability of being predicted as a **safe loan**.


```python
predictions_prob = model_5.predict_proba(validation_X)

validation_data['predictions'] = 1- predictions_prob
validation_data.sort('predictions', ascending=False).head(5)

```

    C:\Users\SSQ\AppData\Roaming\Python\Python27\site-packages\ipykernel\__main__.py:3: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      app.launch_new_instance()
    C:\Users\SSQ\AppData\Roaming\Python\Python27\site-packages\ipykernel\__main__.py:4: FutureWarning: sort(columns=....) is deprecated, use sort_values(by=.....)
    




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>safe_loans</th>
      <th>sub_grade_num</th>
      <th>short_emp</th>
      <th>emp_length_num</th>
      <th>dti</th>
      <th>payment_inc_ratio</th>
      <th>delinq_2yrs</th>
      <th>delinq_2yrs_zero</th>
      <th>inq_last_6mths</th>
      <th>last_delinq_none</th>
      <th>...</th>
      <th>purpose_home_improvement</th>
      <th>purpose_house</th>
      <th>purpose_major_purchase</th>
      <th>purpose_medical</th>
      <th>purpose_moving</th>
      <th>purpose_other</th>
      <th>purpose_small_business</th>
      <th>purpose_vacation</th>
      <th>purpose_wedding</th>
      <th>predictions</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8021</th>
      <td>-1</td>
      <td>0.4</td>
      <td>0</td>
      <td>4</td>
      <td>12.73</td>
      <td>12.16700</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1</td>
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
      <td>0.673059</td>
    </tr>
    <tr>
      <th>19815</th>
      <td>1</td>
      <td>0.6</td>
      <td>0</td>
      <td>11</td>
      <td>2.40</td>
      <td>2.49545</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1</td>
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
      <td>0.661468</td>
    </tr>
    <tr>
      <th>93680</th>
      <td>1</td>
      <td>0.4</td>
      <td>0</td>
      <td>6</td>
      <td>15.39</td>
      <td>6.14080</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1</td>
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
      <td>0.661468</td>
    </tr>
    <tr>
      <th>112700</th>
      <td>1</td>
      <td>0.4</td>
      <td>0</td>
      <td>11</td>
      <td>10.95</td>
      <td>3.02852</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1</td>
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
      <td>0.661468</td>
    </tr>
    <tr>
      <th>104046</th>
      <td>1</td>
      <td>0.4</td>
      <td>0</td>
      <td>2</td>
      <td>9.18</td>
      <td>4.26159</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1</td>
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
      <td>0.661468</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 46 columns</p>
</div>



**Checkpoint:** For each row, the probabilities should be a number in the range **[0, 1]**. We have provided a simple check here to make sure your answers are correct.


```python
print "Your loans      : %s\n" % validation_data.sort('predictions', ascending=False)['predictions'].head(5)
#print "Expected answer : %s" % [0.4492515948736132, 0.6119100103640573,
 #                               0.3835981314851436, 0.3693306705994325]
```

    Your loans      : 8021      0.673059
    19815     0.661468
    93680     0.661468
    112700    0.661468
    104046    0.661468
    Name: predictions, dtype: float64
    
    

    C:\Users\SSQ\AppData\Roaming\Python\Python27\site-packages\ipykernel\__main__.py:1: FutureWarning: sort(columns=....) is deprecated, use sort_values(by=.....)
      if __name__ == '__main__':
    


```python
print validation_data.sort('predictions', ascending=False).head(5)[['grade_A','grade_B', 'grade_C', 'grade_D', 'grade_E', 'grade_F', 'grade_G']]
```

            grade_A  grade_B  grade_C  grade_D  grade_E  grade_F  grade_G
    8021          1        0        0        0        0        0        0
    19815         1        0        0        0        0        0        0
    93680         1        0        0        0        0        0        0
    112700        1        0        0        0        0        0        0
    104046        1        0        0        0        0        0        0
    

    C:\Users\SSQ\AppData\Roaming\Python\Python27\site-packages\ipykernel\__main__.py:1: FutureWarning: sort(columns=....) is deprecated, use sort_values(by=.....)
      if __name__ == '__main__':
    

** Quiz Question**: What grades are the top 5 loans?

Let us repeat this excercise to find the top 5 loans (in the **validation_data**) with the **lowest probability** of being predicted as a **safe loan**:


```python
validation_data.sort('predictions', ascending=True).head(5)
```

    C:\Users\SSQ\AppData\Roaming\Python\Python27\site-packages\ipykernel\__main__.py:1: FutureWarning: sort(columns=....) is deprecated, use sort_values(by=.....)
      if __name__ == '__main__':
    




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>safe_loans</th>
      <th>sub_grade_num</th>
      <th>short_emp</th>
      <th>emp_length_num</th>
      <th>dti</th>
      <th>payment_inc_ratio</th>
      <th>delinq_2yrs</th>
      <th>delinq_2yrs_zero</th>
      <th>inq_last_6mths</th>
      <th>last_delinq_none</th>
      <th>...</th>
      <th>purpose_home_improvement</th>
      <th>purpose_house</th>
      <th>purpose_major_purchase</th>
      <th>purpose_medical</th>
      <th>purpose_moving</th>
      <th>purpose_other</th>
      <th>purpose_small_business</th>
      <th>purpose_vacation</th>
      <th>purpose_wedding</th>
      <th>predictions</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>58794</th>
      <td>-1</td>
      <td>0.8</td>
      <td>0</td>
      <td>2</td>
      <td>8.66</td>
      <td>17.62510</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1</td>
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
      <td>0.307334</td>
    </tr>
    <tr>
      <th>84508</th>
      <td>-1</td>
      <td>0.8</td>
      <td>1</td>
      <td>1</td>
      <td>7.37</td>
      <td>16.62070</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1</td>
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
      <td>0.307334</td>
    </tr>
    <tr>
      <th>27502</th>
      <td>-1</td>
      <td>1.0</td>
      <td>0</td>
      <td>3</td>
      <td>8.53</td>
      <td>14.62800</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.312806</td>
    </tr>
    <tr>
      <th>84921</th>
      <td>-1</td>
      <td>0.8</td>
      <td>0</td>
      <td>9</td>
      <td>8.54</td>
      <td>7.48113</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.315973</td>
    </tr>
    <tr>
      <th>101746</th>
      <td>-1</td>
      <td>0.2</td>
      <td>0</td>
      <td>11</td>
      <td>11.21</td>
      <td>4.23624</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.315973</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 46 columns</p>
</div>




```python
print "Your loans      : %s\n" % validation_data.sort('predictions', ascending=True)['predictions'].head(5)
```

    Your loans      : 58794     0.307334
    84508     0.307334
    27502     0.312806
    84921     0.315973
    101746    0.315973
    Name: predictions, dtype: float64
    
    

    C:\Users\SSQ\AppData\Roaming\Python\Python27\site-packages\ipykernel\__main__.py:1: FutureWarning: sort(columns=....) is deprecated, use sort_values(by=.....)
      if __name__ == '__main__':
    


```python
print validation_data.sort('predictions', ascending=True).head(5)[['grade_A','grade_B', 'grade_C', 'grade_D', 'grade_E', 'grade_F', 'grade_G']]
```

            grade_A  grade_B  grade_C  grade_D  grade_E  grade_F  grade_G
    58794         0        0        1        0        0        0        0
    84508         0        0        1        0        0        0        0
    27502         0        0        1        0        0        0        0
    84921         0        0        1        0        0        0        0
    101746        0        0        0        1        0        0        0
    

    C:\Users\SSQ\AppData\Roaming\Python\Python27\site-packages\ipykernel\__main__.py:1: FutureWarning: sort(columns=....) is deprecated, use sort_values(by=.....)
      if __name__ == '__main__':
    

**Checkpoint:** You should expect to see 5 loans with the grade ['**D**', '**C**', '**C**', '**C**', '**B**'] or with ['**D**', '**C**', '**B**', '**C**', '**C**'].

## Effect of adding more trees

In this assignment, we will train 5 different ensemble classifiers in the form of gradient boosted trees. We will train models with 10, 50, 100, 200, and 500 trees.  We use the **max_iterations** parameter in the boosted tree module. 

Let's get sarted with a model with **max_iterations = 10**:


```python
model_10 = GradientBoostingClassifier(n_estimators=10, max_depth=6).fit(train_X, train_Y)
```

Now, train 4 models with **max_iterations** to be:
* `max_iterations = 50`, 
* `max_iterations = 100`
* `max_iterations = 200`
* `max_iterations = 500`. 

Let us call these models **model_50**, **model_100**, **model_200**, and **model_500**. You can pass in `verbose=False` in order to suppress the printed output.

**Warning:** This could take a couple of minutes to run.


```python
model_50 = GradientBoostingClassifier(n_estimators=50, max_depth=6).fit(train_X, train_Y)
model_100 = GradientBoostingClassifier(n_estimators=100, max_depth=6).fit(train_X, train_Y)
model_200 = GradientBoostingClassifier(n_estimators=200, max_depth=6).fit(train_X, train_Y)
model_500 = GradientBoostingClassifier(n_estimators=500, max_depth=6).fit(train_X, train_Y)
```

## Compare accuracy on entire validation set

Now we will compare the predicitve accuracy of our models on the validation set. Evaluate the **accuracy** of the 10, 50, 100, 200, and 500 tree models on the **validation_data**. Use the `.evaluate` method.


```python
print model_10.score(validation_X,validation_Y)
print model_50.score(validation_X,validation_Y)
print model_100.score(validation_X,validation_Y)
print model_200.score(validation_X,validation_Y)
print model_500.score(validation_X,validation_Y)
```

    0.666307626023
    0.685264971995
    0.690219732874
    0.68935803533
    0.687203791469
    

**Quiz Question:** Which model has the **best** accuracy on the **validation_data**?

**model_100**

**Quiz Question:** Is it always true that the model with the most trees will perform best on test data?

**NO**

## Plot the training and validation error vs. number of trees

Recall from the lecture that the classification error is defined as

$$
\mbox{classification error} = 1 - \mbox{accuracy} 
$$

In this section, we will plot the **training and validation errors versus the number of trees** to get a sense of how these models are performing. We will compare the 10, 50, 100, 200, and 500 tree models. You will need [matplotlib](http://matplotlib.org/downloads.html) in order to visualize the plots. 

First, make sure this block of code runs on your computer.


```python
import matplotlib.pyplot as plt
%matplotlib inline
def make_figure(dim, title, xlabel, ylabel, legend):
    plt.rcParams['figure.figsize'] = dim
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if legend is not None:
        plt.legend(loc=legend, prop={'size':15})
    plt.rcParams.update({'font.size': 16})
    plt.tight_layout()
```

In order to plot the classification errors (on the **train_data** and **validation_data**) versus the number of trees, we will need lists of these accuracies, which we get by applying the method `.evaluate`. 

**Steps to follow:**

* **Step 1:** Calculate the classification error for model on the training data (**train_data**).
* **Step 2:** Store the training errors into a list (called `training_errors`) that looks like this:
```
[train_err_10, train_err_50, ..., train_err_500]
```
* **Step 3:** Calculate the classification error of each model on the validation data (**validation_data**).
* **Step 4:** Store the validation classification error into a list (called `validation_errors`) that looks like this:
```
[validation_err_10, validation_err_50, ..., validation_err_500]
```
Once that has been completed, the rest of the code should be able to evaluate correctly and generate the plot.


Let us start with **Step 1**. Write code to compute the classification error on the **train_data** for models **model_10**, **model_50**, **model_100**, **model_200**, and **model_500**.


```python
train_err_10 = 1-model_10.score(train_X,train_Y)
train_err_50 = 1-model_50.score(train_X,train_Y)
train_err_100 = 1-model_100.score(train_X,train_Y)
train_err_200 = 1-model_200.score(train_X,train_Y)
train_err_500 = 1-model_500.score(train_X,train_Y)
```

Now, let us run **Step 2**. Save the training errors into a list called **training_errors**


```python
training_errors = [train_err_10, train_err_50, train_err_100, 
                   train_err_200, train_err_500]
```

Now, onto **Step 3**. Write code to compute the classification error on the **validation_data** for models **model_10**, **model_50**, **model_100**, **model_200**, and **model_500**.


```python
validation_err_10 = 1-model_10.score(validation_X,validation_Y)
validation_err_50 = 1-model_50.score(validation_X,validation_Y)
validation_err_100 = 1-model_100.score(validation_X,validation_Y)
validation_err_200 = 1-model_200.score(validation_X,validation_Y)
validation_err_500 = 1-model_500.score(validation_X,validation_Y)
```

Now, let us run **Step 4**. Save the training errors into a list called **validation_errors**


```python
validation_errors = [validation_err_10, validation_err_50, validation_err_100, 
                     validation_err_200, validation_err_500]
```

Now, we will plot the **training_errors** and **validation_errors** versus the number of trees. We will compare the 10, 50, 100, 200, and 500 tree models. We provide some plotting code to visualize the plots within this notebook. 

Run the following code to visualize the plots.


```python
plt.plot([10, 50, 100, 200, 500], training_errors, linewidth=4.0, label='Training error')
plt.plot([10, 50, 100, 200, 500], validation_errors, linewidth=4.0, label='Validation error')

make_figure(dim=(10,5), title='Error vs number of trees',
            xlabel='Number of trees',
            ylabel='Classification error',
            legend='best')
```


![png](output_80_0.png)


**Quiz Question**: Does the training error reduce as the number of trees increases?

**Yes**

**Quiz Question**: Is it always true that the validation error will reduce as the number of trees increases?

**NO**


```python

```
