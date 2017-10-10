
# Decision Trees in Practice

In this assignment we will explore various techniques for preventing overfitting in decision trees. We will extend the implementation of the binary decision trees that we implemented in the previous assignment. You will have to use your solutions from this previous assignment and extend them.

In this assignment you will:

* Implement binary decision trees with different early stopping methods.
* Compare models with different stopping parameters.
* Visualize the concept of overfitting in decision trees.

Let's get started!


```python
import numpy as np
import pandas as pd
import json
```

# Load LendingClub Dataset

This assignment will use the [LendingClub](https://www.lendingclub.com/) dataset used in the previous two assignments.


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



As before, we reassign the labels to have +1 for a safe loan, and -1 for a risky (bad) loan.


```python
loans['safe_loans'] = loans['bad_loans'].apply(lambda x : +1 if x==0 else -1)
loans = loans.drop('bad_loans', axis=1)
```

We will be using the same 4 categorical features as in the previous assignment: 
1. grade of the loan 
2. the length of the loan term
3. the home ownership status: own, mortgage, rent
4. number of years of employment.

In the dataset, each of these features is a categorical feature. Since we are building a binary decision tree, we will have to convert this to binary data in a subsequent section using 1-hot encoding.


```python
features = ['grade',              # grade of the loan
            'term',               # the term of the loan
            'home_ownership',     # home_ownership status: own, mortgage or rent
            'emp_length',         # number of years of employment
           ]
target = 'safe_loans'
loans = loans[features + [target]]
```

## Transform categorical data into binary features

Since we are implementing **binary decision trees**, we transform our categorical data into binary data using 1-hot encoding, just as in the previous assignment. Here is the summary of that discussion:

For instance, the **home_ownership** feature represents the home ownership status of the loanee, which is either `own`, `mortgage` or `rent`. For example, if a data point has the feature 
```
   {'home_ownership': 'RENT'}
```
we want to turn this into three features: 
```
 { 
   'home_ownership = OWN'      : 0, 
   'home_ownership = MORTGAGE' : 0, 
   'home_ownership = RENT'     : 1
 }
```

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

       safe_loans  grade_A  grade_B  grade_C  grade_D  grade_E  grade_F  grade_G  \
    0           1        0        1        0        0        0        0        0   
    1          -1        0        0        1        0        0        0        0   
    
       term_ 36 months  term_ 60 months       ...        emp_length_2 years  \
    0                1                0       ...                         0   
    1                0                1       ...                         0   
    
       emp_length_3 years  emp_length_4 years  emp_length_5 years  \
    0                   0                   0                   0   
    1                   0                   0                   0   
    
       emp_length_6 years  emp_length_7 years  emp_length_8 years  \
    0                   0                   0                   0   
    1                   0                   0                   0   
    
       emp_length_9 years  emp_length_< 1 year  emp_length_n/a  
    0                   0                    0               0  
    1                   0                    1               0  
    
    [2 rows x 26 columns]
    Index([u'safe_loans', u'grade_A', u'grade_B', u'grade_C', u'grade_D',
           u'grade_E', u'grade_F', u'grade_G', u'term_ 36 months',
           u'term_ 60 months', u'home_ownership_MORTGAGE', u'home_ownership_OTHER',
           u'home_ownership_OWN', u'home_ownership_RENT', u'emp_length_1 year',
           u'emp_length_10+ years', u'emp_length_2 years', u'emp_length_3 years',
           u'emp_length_4 years', u'emp_length_5 years', u'emp_length_6 years',
           u'emp_length_7 years', u'emp_length_8 years', u'emp_length_9 years',
           u'emp_length_< 1 year', u'emp_length_n/a'],
          dtype='object')
    


```python
loans.iloc[122602]
```




    safe_loans                -1
    grade_A                    0
    grade_B                    0
    grade_C                    0
    grade_D                    0
    grade_E                    1
    grade_F                    0
    grade_G                    0
    term_ 36 months            0
    term_ 60 months            1
    home_ownership_MORTGAGE    1
    home_ownership_OTHER       0
    home_ownership_OWN         0
    home_ownership_RENT        0
    emp_length_1 year          0
    emp_length_10+ years       0
    emp_length_2 years         0
    emp_length_3 years         0
    emp_length_4 years         0
    emp_length_5 years         0
    emp_length_6 years         0
    emp_length_7 years         0
    emp_length_8 years         0
    emp_length_9 years         0
    emp_length_< 1 year        0
    emp_length_n/a             1
    Name: 122602, dtype: int64



## Train-Validation split

We split the data into a train-validation split with 80% of the data in the training set and 20% of the data in the validation set. We use `seed=1` so that everyone gets the same result.


```python
with open('module-6-assignment-train-idx.json') as train_data_file:    
    train_idx  = json.load(train_data_file)
with open('module-6-assignment-validation-idx.json') as validation_data_file:    
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

    37224
    9284
    


```python
train_data = loans.iloc[train_idx]
validation_data = loans.iloc[validation_idx]
```


```python
print len(loans.dtypes )

```

    26
    

# Early stopping methods for decision trees

In this section, we will extend the **binary tree implementation** from the previous assignment in order to handle some early stopping conditions. Recall the 3 early stopping methods that were discussed in lecture:

1. Reached a **maximum depth**. (set by parameter `max_depth`).
2. Reached a **minimum node size**. (set by parameter `min_node_size`).
3. Don't split if the **gain in error reduction** is too small. (set by parameter `min_error_reduction`).

For the rest of this assignment, we will refer to these three as **early stopping conditions 1, 2, and 3**.

## Early stopping condition 1: Maximum depth

Recall that we already implemented the maximum depth stopping condition in the previous assignment. In this assignment, we will experiment with this condition a bit more and also write code to implement the 2nd and 3rd early stopping conditions.

We will be reusing code from the previous assignment and then building upon this.  We will **alert you** when you reach a function that was part of the previous assignment so that you can simply copy and past your previous code.

## Early stopping condition 2: Minimum node size

The function **reached_minimum_node_size** takes 2 arguments:

1. The `data` (from a node)
2. The minimum number of data points that a node is allowed to split on, `min_node_size`.

This function simply calculates whether the number of data points at a given node is less than or equal to the specified minimum node size. This function will be used to detect this early stopping condition in the **decision_tree_create** function.

Fill in the parts of the function below where you find `## YOUR CODE HERE`.  There is **one** instance in the function below.


```python
def reached_minimum_node_size(data, min_node_size):
    # Return True if the number of data points is less than or equal to the minimum node size.
    ## YOUR CODE HERE
    if len(data) <= min_node_size:
        return True
    else:
        return False
```

** Quiz Question:** Given an intermediate node with 6 safe loans and 3 risky loans, if the `min_node_size` parameter is 10, what should the tree learning algorithm do next?

STOP

## Early stopping condition 3: Minimum gain in error reduction

The function **error_reduction** takes 2 arguments:

1. The error **before** a split, `error_before_split`.
2. The error **after** a split, `error_after_split`.

This function computes the gain in error reduction, i.e., the difference between the error before the split and that after the split. This function will be used to detect this early stopping condition in the **decision_tree_create** function.

Fill in the parts of the function below where you find `## YOUR CODE HERE`.  There is **one** instance in the function below. 


```python
def error_reduction(error_before_split, error_after_split):
    # Return the error before the split minus the error after the split.
    ## YOUR CODE HERE
    return error_before_split - error_after_split

```

** Quiz Question:** Assume an intermediate node has 6 safe loans and 3 risky loans.  For each of 4 possible features to split on, the error reduction is 0.0, 0.05, 0.1, and 0.14, respectively. If the **minimum gain in error reduction** parameter is set to 0.2, what should the tree learning algorithm do next?

STOP

## Grabbing binary decision tree helper functions from past assignment

Recall from the previous assignment that we wrote a function `intermediate_node_num_mistakes` that calculates the number of **misclassified examples** when predicting the **majority class**. This is used to help determine which feature is best to split on at a given node of the tree.

**Please copy and paste your code for `intermediate_node_num_mistakes` here**.


```python
def intermediate_node_num_mistakes(labels_in_node):
    # Corner case: If labels_in_node is empty, return 0
    if len(labels_in_node) == 0:
        return 0    
    # Count the number of 1's (safe loans)
    ## YOUR CODE HERE    
    safe_loan = (labels_in_node==1).sum()
    # Count the number of -1's (risky loans)
    ## YOUR CODE HERE                
    risky_loan = (labels_in_node==-1).sum()
    # Return the number of mistakes that the majority classifier makes.
    ## YOUR CODE HERE    
    return min(safe_loan, risky_loan)

```

We then wrote a function `best_splitting_feature` that finds the best feature to split on given the data and a list of features to consider.

**Please copy and paste your `best_splitting_feature` code here**.


```python
def best_splitting_feature(data, features, target):
    
    target_values = data[target]
    best_feature = None # Keep track of the best feature 
    best_error = 10     # Keep track of the best error so far 
    # Note: Since error is always <= 1, we should intialize it with something larger than 1.

    # Convert to float to make sure error gets computed correctly.
    num_data_points = float(len(data))  
    
    # Loop through each feature to consider splitting on that feature
    for feature in features:
        
        # The left split will have all data points where the feature value is 0
        left_split = data[data[feature] == 0]
        
        # The right split will have all data points where the feature value is 1
        ## YOUR CODE HERE
        right_split = data[data[feature] == 1]
            
        # Calculate the number of misclassified examples in the left split.
        # Remember that we implemented a function for this! (It was called intermediate_node_num_mistakes)
        # YOUR CODE HERE
        left_mistakes = intermediate_node_num_mistakes(left_split[target])            

        # Calculate the number of misclassified examples in the right split.
        ## YOUR CODE HERE
        right_mistakes = intermediate_node_num_mistakes(right_split[target])  
            
        # Compute the classification error of this split.
        # Error = (# of mistakes (left) + # of mistakes (right)) / (# of data points)
        ## YOUR CODE HERE
        error = (left_mistakes + right_mistakes) / num_data_points

        # If this is the best error we have found so far, store the feature as best_feature and the error as best_error
        ## YOUR CODE HERE
        if error < best_error:
            best_feature = feature
            best_error = error
    
    return best_feature # Return the best feature we found
```

Finally, recall the function `create_leaf` from the previous assignment, which creates a leaf node given a set of target values.  

**Please copy and paste your `create_leaf` code here**.


```python
def create_leaf(target_values):    
    # Create a leaf node
    leaf = {'splitting_feature' : None,
            'left' : None,
            'right' : None,
            'is_leaf': True }   ## YOUR CODE HERE 
   
    # Count the number of data points that are +1 and -1 in this node.
    num_ones = len(target_values[target_values == +1])
    num_minus_ones = len(target_values[target_values == -1])    

    # For the leaf node, set the prediction to be the majority class.
    # Store the predicted class (1 or -1) in leaf['prediction']
    if num_ones > num_minus_ones:
        leaf['prediction'] = 1         ## YOUR CODE HERE
    else:
        leaf['prediction'] = -1         ## YOUR CODE HERE        

    # Return the leaf node
    return leaf 
```

## Incorporating new early stopping conditions in binary decision tree implementation

Now, you will implement a function that builds a decision tree handling the three early stopping conditions described in this assignment.  In particular, you will write code to detect early stopping conditions 2 and 3.  You implemented above the functions needed to detect these conditions.  The 1st early stopping condition, **max_depth**, was implemented in the previous assigment and you will not need to reimplement this.  In addition to these early stopping conditions, the typical stopping conditions of having no mistakes or no more features to split on (which we denote by "stopping conditions" 1 and 2) are also included as in the previous assignment.

**Implementing early stopping condition 2: minimum node size:**

* **Step 1:** Use the function **reached_minimum_node_size** that you implemented earlier to write an if condition to detect whether we have hit the base case, i.e., the node does not have enough data points and should be turned into a leaf. Don't forget to use the `min_node_size` argument.
* **Step 2:** Return a leaf. This line of code should be the same as the other (pre-implemented) stopping conditions.


**Implementing early stopping condition 3: minimum error reduction:**

**Note:** This has to come after finding the best splitting feature so we can calculate the error after splitting in order to calculate the error reduction.

* **Step 1:** Calculate the **classification error before splitting**.  Recall that classification error is defined as:

$$
\text{classification error} = \frac{\text{# mistakes}}{\text{# total examples}}
$$
* **Step 2:** Calculate the **classification error after splitting**. This requires calculating the number of mistakes in the left and right splits, and then dividing by the total number of examples.
* **Step 3:** Use the function **error_reduction** to that you implemented earlier to write an if condition to detect whether  the reduction in error is less than the constant provided (`min_error_reduction`). Don't forget to use that argument.
* **Step 4:** Return a leaf. This line of code should be the same as the other (pre-implemented) stopping conditions.

Fill in the places where you find `## YOUR CODE HERE`. There are **seven** places in this function for you to fill in.


```python
def decision_tree_create(data, features, target, current_depth = 0, 
                         max_depth = 10, min_node_size=1, 
                         min_error_reduction=0.0):
    
    remaining_features = features[:] # Make a copy of the features.
    
    target_values = data[target]
    print "--------------------------------------------------------------------"
    print "Subtree, depth = %s (%s data points)." % (current_depth, len(target_values))
    
    
    # Stopping condition 1: All nodes are of the same type.
    if intermediate_node_num_mistakes(target_values) == 0:
        print "Stopping condition 1 reached. All data points have the same target value."                
        return create_leaf(target_values)
    
    # Stopping condition 2: No more features to split on.
    if remaining_features == []:
        print "Stopping condition 2 reached. No remaining features."                
        return create_leaf(target_values)    
    
    # Early stopping condition 1: Reached max depth limit.
    if current_depth >= max_depth:
        print "Early stopping condition 1 reached. Reached maximum depth."
        return create_leaf(target_values)
    
    # Early stopping condition 2: Reached the minimum node size.
    # If the number of data points is less than or equal to the minimum size, return a leaf.
    if  reached_minimum_node_size(data, min_node_size):          ## YOUR CODE HERE 
        print "Early stopping condition 2 reached. Reached minimum node size."
        return  create_leaf(target_values) ## YOUR CODE HERE
    
    # Find the best splitting feature
    splitting_feature = best_splitting_feature(data, features, target)
    
    # Split on the best feature that we found. 
    left_split = data[data[splitting_feature] == 0]
    right_split = data[data[splitting_feature] == 1]
    
    # Early stopping condition 3: Minimum error reduction
    # Calculate the error before splitting (number of misclassified examples 
    # divided by the total number of examples)
    error_before_split = intermediate_node_num_mistakes(target_values) / float(len(data))
    
    # Calculate the error after splitting (number of misclassified examples 
    # in both groups divided by the total number of examples)
    left_mistakes = intermediate_node_num_mistakes(left_split[target])   ## YOUR CODE HERE
    right_mistakes = intermediate_node_num_mistakes(right_split[target])  ## YOUR CODE HERE
    error_after_split = (left_mistakes + right_mistakes) / float(len(data))
    
    # If the error reduction is LESS THAN OR EQUAL TO min_error_reduction, return a leaf.
    if error_reduction(error_before_split, error_after_split) <= min_error_reduction:       ## YOUR CODE HERE
        print "Early stopping condition 3 reached. Minimum error reduction."
        return create_leaf(target_values)  ## YOUR CODE HERE 
    
    
    remaining_features.remove(splitting_feature)
    print "Split on feature %s. (%s, %s)" % (\
                      splitting_feature, len(left_split), len(right_split))
    
    
    # Repeat (recurse) on left and right subtrees
    left_tree = decision_tree_create(left_split, remaining_features, target, 
                                     current_depth + 1, max_depth, min_node_size, min_error_reduction)        
    
    ## YOUR CODE HERE
    right_tree = decision_tree_create(right_split, remaining_features, target, 
                                     current_depth + 1, max_depth, min_node_size, min_error_reduction)    
    
    
    return {'is_leaf'          : False, 
            'prediction'       : None,
            'splitting_feature': splitting_feature,
            'left'             : left_tree, 
            'right'            : right_tree}
```

Here is a function to count the nodes in your tree:


```python
def count_nodes(tree):
    if tree['is_leaf']:
        return 1
    return 1 + count_nodes(tree['left']) + count_nodes(tree['right'])
```


```python
features = list(train_data.columns)
features.remove('safe_loans')

print list(train_data.columns)
print features
```

    ['safe_loans', 'grade_A', 'grade_B', 'grade_C', 'grade_D', 'grade_E', 'grade_F', 'grade_G', 'term_ 36 months', 'term_ 60 months', 'home_ownership_MORTGAGE', 'home_ownership_OTHER', 'home_ownership_OWN', 'home_ownership_RENT', 'emp_length_1 year', 'emp_length_10+ years', 'emp_length_2 years', 'emp_length_3 years', 'emp_length_4 years', 'emp_length_5 years', 'emp_length_6 years', 'emp_length_7 years', 'emp_length_8 years', 'emp_length_9 years', 'emp_length_< 1 year', 'emp_length_n/a']
    ['grade_A', 'grade_B', 'grade_C', 'grade_D', 'grade_E', 'grade_F', 'grade_G', 'term_ 36 months', 'term_ 60 months', 'home_ownership_MORTGAGE', 'home_ownership_OTHER', 'home_ownership_OWN', 'home_ownership_RENT', 'emp_length_1 year', 'emp_length_10+ years', 'emp_length_2 years', 'emp_length_3 years', 'emp_length_4 years', 'emp_length_5 years', 'emp_length_6 years', 'emp_length_7 years', 'emp_length_8 years', 'emp_length_9 years', 'emp_length_< 1 year', 'emp_length_n/a']
    

Run the following test code to check your implementation. Make sure you get **'Test passed'** before proceeding.


```python
small_decision_tree = decision_tree_create(train_data, features, 'safe_loans', max_depth = 2, 
                                        min_node_size = 10, min_error_reduction=0.0)
if count_nodes(small_decision_tree) == 7:
    print 'Test passed!'
else:
    print 'Test failed... try again!'
    print 'Number of nodes found                :', count_nodes(small_decision_tree)
    print 'Number of nodes that should be there : 7' 
```

    --------------------------------------------------------------------
    Subtree, depth = 0 (37224 data points).
    Split on feature term_ 36 months. (9223, 28001)
    --------------------------------------------------------------------
    Subtree, depth = 1 (9223 data points).
    Split on feature grade_A. (9122, 101)
    --------------------------------------------------------------------
    Subtree, depth = 2 (9122 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 2 (101 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 1 (28001 data points).
    Split on feature grade_D. (23300, 4701)
    --------------------------------------------------------------------
    Subtree, depth = 2 (23300 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 2 (4701 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    Test passed!
    

## Build a tree!

Now that your code is working, we will train a tree model on the **train_data** with
* `max_depth = 6`
* `min_node_size = 100`, 
* `min_error_reduction = 0.0`

**Warning**: This code block may take a minute to learn. 


```python
my_decision_tree_new = decision_tree_create(train_data, features, 'safe_loans', max_depth = 6, 
                                min_node_size = 100, min_error_reduction=0.0)
```

    --------------------------------------------------------------------
    Subtree, depth = 0 (37224 data points).
    Split on feature term_ 36 months. (9223, 28001)
    --------------------------------------------------------------------
    Subtree, depth = 1 (9223 data points).
    Split on feature grade_A. (9122, 101)
    --------------------------------------------------------------------
    Subtree, depth = 2 (9122 data points).
    Early stopping condition 3 reached. Minimum error reduction.
    --------------------------------------------------------------------
    Subtree, depth = 2 (101 data points).
    Split on feature emp_length_n/a. (96, 5)
    --------------------------------------------------------------------
    Subtree, depth = 3 (96 data points).
    Early stopping condition 2 reached. Reached minimum node size.
    --------------------------------------------------------------------
    Subtree, depth = 3 (5 data points).
    Early stopping condition 2 reached. Reached minimum node size.
    --------------------------------------------------------------------
    Subtree, depth = 1 (28001 data points).
    Split on feature grade_D. (23300, 4701)
    --------------------------------------------------------------------
    Subtree, depth = 2 (23300 data points).
    Split on feature grade_E. (22024, 1276)
    --------------------------------------------------------------------
    Subtree, depth = 3 (22024 data points).
    Split on feature grade_F. (21666, 358)
    --------------------------------------------------------------------
    Subtree, depth = 4 (21666 data points).
    Split on feature emp_length_n/a. (20734, 932)
    --------------------------------------------------------------------
    Subtree, depth = 5 (20734 data points).
    Split on feature grade_G. (20638, 96)
    --------------------------------------------------------------------
    Subtree, depth = 6 (20638 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (96 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 5 (932 data points).
    Split on feature grade_A. (702, 230)
    --------------------------------------------------------------------
    Subtree, depth = 6 (702 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (230 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 4 (358 data points).
    Split on feature emp_length_8 years. (347, 11)
    --------------------------------------------------------------------
    Subtree, depth = 5 (347 data points).
    Early stopping condition 3 reached. Minimum error reduction.
    --------------------------------------------------------------------
    Subtree, depth = 5 (11 data points).
    Early stopping condition 2 reached. Reached minimum node size.
    --------------------------------------------------------------------
    Subtree, depth = 3 (1276 data points).
    Early stopping condition 3 reached. Minimum error reduction.
    --------------------------------------------------------------------
    Subtree, depth = 2 (4701 data points).
    Early stopping condition 3 reached. Minimum error reduction.
    

Let's now train a tree model **ignoring early stopping conditions 2 and 3** so that we get the same tree as in the previous assignment.  To ignore these conditions, we set `min_node_size=0` and `min_error_reduction=-1` (a negative value).


```python
my_decision_tree_old = decision_tree_create(train_data, features, 'safe_loans', max_depth = 6, 
                                min_node_size = 0, min_error_reduction=-1)
```

    --------------------------------------------------------------------
    Subtree, depth = 0 (37224 data points).
    Split on feature term_ 36 months. (9223, 28001)
    --------------------------------------------------------------------
    Subtree, depth = 1 (9223 data points).
    Split on feature grade_A. (9122, 101)
    --------------------------------------------------------------------
    Subtree, depth = 2 (9122 data points).
    Split on feature grade_B. (8074, 1048)
    --------------------------------------------------------------------
    Subtree, depth = 3 (8074 data points).
    Split on feature grade_C. (5884, 2190)
    --------------------------------------------------------------------
    Subtree, depth = 4 (5884 data points).
    Split on feature grade_D. (3826, 2058)
    --------------------------------------------------------------------
    Subtree, depth = 5 (3826 data points).
    Split on feature grade_E. (1693, 2133)
    --------------------------------------------------------------------
    Subtree, depth = 6 (1693 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (2133 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 5 (2058 data points).
    Split on feature grade_E. (2058, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (2058 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 4 (2190 data points).
    Split on feature grade_D. (2190, 0)
    --------------------------------------------------------------------
    Subtree, depth = 5 (2190 data points).
    Split on feature grade_E. (2190, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (2190 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 3 (1048 data points).
    Split on feature emp_length_5 years. (969, 79)
    --------------------------------------------------------------------
    Subtree, depth = 4 (969 data points).
    Split on feature grade_C. (969, 0)
    --------------------------------------------------------------------
    Subtree, depth = 5 (969 data points).
    Split on feature grade_D. (969, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (969 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 4 (79 data points).
    Split on feature home_ownership_MORTGAGE. (34, 45)
    --------------------------------------------------------------------
    Subtree, depth = 5 (34 data points).
    Split on feature grade_C. (34, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (34 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (45 data points).
    Split on feature grade_C. (45, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (45 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 2 (101 data points).
    Split on feature emp_length_n/a. (96, 5)
    --------------------------------------------------------------------
    Subtree, depth = 3 (96 data points).
    Split on feature emp_length_< 1 year. (85, 11)
    --------------------------------------------------------------------
    Subtree, depth = 4 (85 data points).
    Split on feature grade_B. (85, 0)
    --------------------------------------------------------------------
    Subtree, depth = 5 (85 data points).
    Split on feature grade_C. (85, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (85 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 4 (11 data points).
    Split on feature grade_B. (11, 0)
    --------------------------------------------------------------------
    Subtree, depth = 5 (11 data points).
    Split on feature grade_C. (11, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (11 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 3 (5 data points).
    Split on feature grade_B. (5, 0)
    --------------------------------------------------------------------
    Subtree, depth = 4 (5 data points).
    Split on feature grade_C. (5, 0)
    --------------------------------------------------------------------
    Subtree, depth = 5 (5 data points).
    Split on feature grade_D. (5, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (5 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 4 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 1 (28001 data points).
    Split on feature grade_D. (23300, 4701)
    --------------------------------------------------------------------
    Subtree, depth = 2 (23300 data points).
    Split on feature grade_E. (22024, 1276)
    --------------------------------------------------------------------
    Subtree, depth = 3 (22024 data points).
    Split on feature grade_F. (21666, 358)
    --------------------------------------------------------------------
    Subtree, depth = 4 (21666 data points).
    Split on feature emp_length_n/a. (20734, 932)
    --------------------------------------------------------------------
    Subtree, depth = 5 (20734 data points).
    Split on feature grade_G. (20638, 96)
    --------------------------------------------------------------------
    Subtree, depth = 6 (20638 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (96 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 5 (932 data points).
    Split on feature grade_A. (702, 230)
    --------------------------------------------------------------------
    Subtree, depth = 6 (702 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (230 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 4 (358 data points).
    Split on feature emp_length_8 years. (347, 11)
    --------------------------------------------------------------------
    Subtree, depth = 5 (347 data points).
    Split on feature grade_A. (347, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (347 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (11 data points).
    Split on feature home_ownership_OWN. (9, 2)
    --------------------------------------------------------------------
    Subtree, depth = 6 (9 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (2 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 3 (1276 data points).
    Split on feature grade_A. (1276, 0)
    --------------------------------------------------------------------
    Subtree, depth = 4 (1276 data points).
    Split on feature grade_B. (1276, 0)
    --------------------------------------------------------------------
    Subtree, depth = 5 (1276 data points).
    Split on feature grade_C. (1276, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (1276 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 4 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 2 (4701 data points).
    Split on feature grade_A. (4701, 0)
    --------------------------------------------------------------------
    Subtree, depth = 3 (4701 data points).
    Split on feature grade_B. (4701, 0)
    --------------------------------------------------------------------
    Subtree, depth = 4 (4701 data points).
    Split on feature grade_C. (4701, 0)
    --------------------------------------------------------------------
    Subtree, depth = 5 (4701 data points).
    Split on feature grade_E. (4701, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (4701 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 4 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 3 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    

## Making predictions

Recall that in the previous assignment you implemented a function `classify` to classify a new point `x` using a given `tree`.

**Please copy and paste your `classify` code here**.


```python
def classify(tree, x, annotate = False):
    # if the node is a leaf node.
    if tree['is_leaf']:
        if annotate:
            print "At leaf, predicting %s" % tree['prediction']
        return tree['prediction']
    else:
        # split on feature.
        split_feature_value = x[tree['splitting_feature']]
        if annotate:
            print "Split on %s = %s" % (tree['splitting_feature'], split_feature_value)
        if split_feature_value == 0:
            return classify(tree['left'], x, annotate)
        else:
            ### YOUR CODE HERE
            return classify(tree['right'], x, annotate)
```

Now, let's consider the first example of the validation set and see what the `my_decision_tree_new` model predicts for this data point.


```python
validation_data.iloc[0]
```




    safe_loans                -1
    grade_A                    0
    grade_B                    0
    grade_C                    0
    grade_D                    1
    grade_E                    0
    grade_F                    0
    grade_G                    0
    term_ 36 months            0
    term_ 60 months            1
    home_ownership_MORTGAGE    0
    home_ownership_OTHER       0
    home_ownership_OWN         0
    home_ownership_RENT        1
    emp_length_1 year          0
    emp_length_10+ years       0
    emp_length_2 years         1
    emp_length_3 years         0
    emp_length_4 years         0
    emp_length_5 years         0
    emp_length_6 years         0
    emp_length_7 years         0
    emp_length_8 years         0
    emp_length_9 years         0
    emp_length_< 1 year        0
    emp_length_n/a             0
    Name: 24, dtype: int64




```python
print 'Predicted class: %s ' % classify(my_decision_tree_new, validation_data.iloc[0])
```

    Predicted class: -1 
    

Let's add some annotations to our prediction to see what the prediction path was that lead to this predicted class:


```python
classify(my_decision_tree_new, validation_data.iloc[0], annotate = True)
```

    Split on term_ 36 months = 0
    Split on grade_A = 0
    At leaf, predicting -1
    




    -1



Let's now recall the prediction path for the decision tree learned in the previous assignment, which we recreated here as `my_decision_tree_old`.


```python
classify(my_decision_tree_old, validation_data.iloc[0], annotate = True)
```

    Split on term_ 36 months = 0
    Split on grade_A = 0
    Split on grade_B = 0
    Split on grade_C = 0
    Split on grade_D = 1
    Split on grade_E = 0
    At leaf, predicting -1
    




    -1



** Quiz Question:** For `my_decision_tree_new` trained with `max_depth = 6`, `min_node_size = 100`, `min_error_reduction=0.0`, is the prediction path for `validation_set[0]` shorter, longer, or the same as for `my_decision_tree_old` that ignored the early stopping conditions 2 and 3?

shorter

**Quiz Question:** For `my_decision_tree_new` trained with `max_depth = 6`, `min_node_size = 100`, `min_error_reduction=0.0`, is the prediction path for **any point** always shorter, always longer, always the same, shorter or the same, or longer or the same as for `my_decision_tree_old` that ignored the early stopping conditions 2 and 3?

shorter or the same

** Quiz Question:** For a tree trained on **any** dataset using `max_depth = 6`, `min_node_size = 100`, `min_error_reduction=0.0`, what is the maximum number of splits encountered while making a single prediction?

6

## Evaluating the model

Now let us evaluate the model that we have trained. You implemented this evaluation in the function `evaluate_classification_error` from the previous assignment.

**Please copy and paste your `evaluate_classification_error` code here**.


```python
def evaluate_classification_error(tree, data, target):
    # Apply the classify(tree, x) to each row in your data
    prediction = data.apply(lambda x: classify(tree, x), axis=1)
    
    # Once you've made the predictions, calculate the classification error and return it
    ## YOUR CODE HERE
    return (data[target] != np.array(prediction)).values.sum() / float(len(data))
```

Now, let's use this function to evaluate the classification error of `my_decision_tree_new` on the **validation_set**.


```python
evaluate_classification_error(my_decision_tree_new, validation_data, target)
```




    0.38367083153813014



Now, evaluate the validation error using `my_decision_tree_old`.


```python
evaluate_classification_error(my_decision_tree_old, validation_data, target)
```




    0.38377854373115039



**Quiz Question:** Is the validation error of the new decision tree (using early stopping conditions 2 and 3) lower than, higher than, or the same as that of the old decision tree from the previous assignment?

lower

# Exploring the effect of max_depth

We will compare three models trained with different values of the stopping criterion. We intentionally picked models at the extreme ends (**too small**, **just right**, and **too large**).

Train three models with these parameters:

1. **model_1**: max_depth = 2 (too small)
2. **model_2**: max_depth = 6 (just right)
3. **model_3**: max_depth = 14 (may be too large)

For each of these three, we set `min_node_size = 0` and `min_error_reduction = -1`.

** Note:** Each tree can take up to a few minutes to train. In particular, `model_3` will probably take the longest to train.


```python
model_1 = decision_tree_create(train_data, features, 'safe_loans', max_depth = 2, 
                                min_node_size = 0, min_error_reduction=-1)
model_2 = decision_tree_create(train_data, features, 'safe_loans', max_depth = 6, 
                                min_node_size = 0, min_error_reduction=-1)
model_3 = decision_tree_create(train_data, features, 'safe_loans', max_depth = 14, 
                                min_node_size = 0, min_error_reduction=-1)
```

    --------------------------------------------------------------------
    Subtree, depth = 0 (37224 data points).
    Split on feature term_ 36 months. (9223, 28001)
    --------------------------------------------------------------------
    Subtree, depth = 1 (9223 data points).
    Split on feature grade_A. (9122, 101)
    --------------------------------------------------------------------
    Subtree, depth = 2 (9122 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 2 (101 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 1 (28001 data points).
    Split on feature grade_D. (23300, 4701)
    --------------------------------------------------------------------
    Subtree, depth = 2 (23300 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 2 (4701 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 0 (37224 data points).
    Split on feature term_ 36 months. (9223, 28001)
    --------------------------------------------------------------------
    Subtree, depth = 1 (9223 data points).
    Split on feature grade_A. (9122, 101)
    --------------------------------------------------------------------
    Subtree, depth = 2 (9122 data points).
    Split on feature grade_B. (8074, 1048)
    --------------------------------------------------------------------
    Subtree, depth = 3 (8074 data points).
    Split on feature grade_C. (5884, 2190)
    --------------------------------------------------------------------
    Subtree, depth = 4 (5884 data points).
    Split on feature grade_D. (3826, 2058)
    --------------------------------------------------------------------
    Subtree, depth = 5 (3826 data points).
    Split on feature grade_E. (1693, 2133)
    --------------------------------------------------------------------
    Subtree, depth = 6 (1693 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (2133 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 5 (2058 data points).
    Split on feature grade_E. (2058, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (2058 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 4 (2190 data points).
    Split on feature grade_D. (2190, 0)
    --------------------------------------------------------------------
    Subtree, depth = 5 (2190 data points).
    Split on feature grade_E. (2190, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (2190 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 3 (1048 data points).
    Split on feature emp_length_5 years. (969, 79)
    --------------------------------------------------------------------
    Subtree, depth = 4 (969 data points).
    Split on feature grade_C. (969, 0)
    --------------------------------------------------------------------
    Subtree, depth = 5 (969 data points).
    Split on feature grade_D. (969, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (969 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 4 (79 data points).
    Split on feature home_ownership_MORTGAGE. (34, 45)
    --------------------------------------------------------------------
    Subtree, depth = 5 (34 data points).
    Split on feature grade_C. (34, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (34 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (45 data points).
    Split on feature grade_C. (45, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (45 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 2 (101 data points).
    Split on feature emp_length_n/a. (96, 5)
    --------------------------------------------------------------------
    Subtree, depth = 3 (96 data points).
    Split on feature emp_length_< 1 year. (85, 11)
    --------------------------------------------------------------------
    Subtree, depth = 4 (85 data points).
    Split on feature grade_B. (85, 0)
    --------------------------------------------------------------------
    Subtree, depth = 5 (85 data points).
    Split on feature grade_C. (85, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (85 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 4 (11 data points).
    Split on feature grade_B. (11, 0)
    --------------------------------------------------------------------
    Subtree, depth = 5 (11 data points).
    Split on feature grade_C. (11, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (11 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 3 (5 data points).
    Split on feature grade_B. (5, 0)
    --------------------------------------------------------------------
    Subtree, depth = 4 (5 data points).
    Split on feature grade_C. (5, 0)
    --------------------------------------------------------------------
    Subtree, depth = 5 (5 data points).
    Split on feature grade_D. (5, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (5 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 4 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 1 (28001 data points).
    Split on feature grade_D. (23300, 4701)
    --------------------------------------------------------------------
    Subtree, depth = 2 (23300 data points).
    Split on feature grade_E. (22024, 1276)
    --------------------------------------------------------------------
    Subtree, depth = 3 (22024 data points).
    Split on feature grade_F. (21666, 358)
    --------------------------------------------------------------------
    Subtree, depth = 4 (21666 data points).
    Split on feature emp_length_n/a. (20734, 932)
    --------------------------------------------------------------------
    Subtree, depth = 5 (20734 data points).
    Split on feature grade_G. (20638, 96)
    --------------------------------------------------------------------
    Subtree, depth = 6 (20638 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (96 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 5 (932 data points).
    Split on feature grade_A. (702, 230)
    --------------------------------------------------------------------
    Subtree, depth = 6 (702 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (230 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 4 (358 data points).
    Split on feature emp_length_8 years. (347, 11)
    --------------------------------------------------------------------
    Subtree, depth = 5 (347 data points).
    Split on feature grade_A. (347, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (347 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (11 data points).
    Split on feature home_ownership_OWN. (9, 2)
    --------------------------------------------------------------------
    Subtree, depth = 6 (9 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (2 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 3 (1276 data points).
    Split on feature grade_A. (1276, 0)
    --------------------------------------------------------------------
    Subtree, depth = 4 (1276 data points).
    Split on feature grade_B. (1276, 0)
    --------------------------------------------------------------------
    Subtree, depth = 5 (1276 data points).
    Split on feature grade_C. (1276, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (1276 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 4 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 2 (4701 data points).
    Split on feature grade_A. (4701, 0)
    --------------------------------------------------------------------
    Subtree, depth = 3 (4701 data points).
    Split on feature grade_B. (4701, 0)
    --------------------------------------------------------------------
    Subtree, depth = 4 (4701 data points).
    Split on feature grade_C. (4701, 0)
    --------------------------------------------------------------------
    Subtree, depth = 5 (4701 data points).
    Split on feature grade_E. (4701, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (4701 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 4 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 3 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 0 (37224 data points).
    Split on feature term_ 36 months. (9223, 28001)
    --------------------------------------------------------------------
    Subtree, depth = 1 (9223 data points).
    Split on feature grade_A. (9122, 101)
    --------------------------------------------------------------------
    Subtree, depth = 2 (9122 data points).
    Split on feature grade_B. (8074, 1048)
    --------------------------------------------------------------------
    Subtree, depth = 3 (8074 data points).
    Split on feature grade_C. (5884, 2190)
    --------------------------------------------------------------------
    Subtree, depth = 4 (5884 data points).
    Split on feature grade_D. (3826, 2058)
    --------------------------------------------------------------------
    Subtree, depth = 5 (3826 data points).
    Split on feature grade_E. (1693, 2133)
    --------------------------------------------------------------------
    Subtree, depth = 6 (1693 data points).
    Split on feature home_ownership_OTHER. (1692, 1)
    --------------------------------------------------------------------
    Subtree, depth = 7 (1692 data points).
    Split on feature grade_F. (339, 1353)
    --------------------------------------------------------------------
    Subtree, depth = 8 (339 data points).
    Split on feature grade_G. (0, 339)
    --------------------------------------------------------------------
    Subtree, depth = 9 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 9 (339 data points).
    Split on feature term_ 60 months. (0, 339)
    --------------------------------------------------------------------
    Subtree, depth = 10 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 10 (339 data points).
    Split on feature home_ownership_MORTGAGE. (175, 164)
    --------------------------------------------------------------------
    Subtree, depth = 11 (175 data points).
    Split on feature home_ownership_OWN. (142, 33)
    --------------------------------------------------------------------
    Subtree, depth = 12 (142 data points).
    Split on feature emp_length_6 years. (133, 9)
    --------------------------------------------------------------------
    Subtree, depth = 13 (133 data points).
    Split on feature home_ownership_RENT. (0, 133)
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 14 (133 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 13 (9 data points).
    Split on feature home_ownership_RENT. (0, 9)
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 14 (9 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 12 (33 data points).
    Split on feature emp_length_n/a. (31, 2)
    --------------------------------------------------------------------
    Subtree, depth = 13 (31 data points).
    Split on feature emp_length_2 years. (30, 1)
    --------------------------------------------------------------------
    Subtree, depth = 14 (30 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (1 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 13 (2 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 11 (164 data points).
    Split on feature emp_length_2 years. (159, 5)
    --------------------------------------------------------------------
    Subtree, depth = 12 (159 data points).
    Split on feature emp_length_3 years. (148, 11)
    --------------------------------------------------------------------
    Subtree, depth = 13 (148 data points).
    Split on feature home_ownership_OWN. (148, 0)
    --------------------------------------------------------------------
    Subtree, depth = 14 (148 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 13 (11 data points).
    Split on feature home_ownership_OWN. (11, 0)
    --------------------------------------------------------------------
    Subtree, depth = 14 (11 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 12 (5 data points).
    Split on feature home_ownership_OWN. (5, 0)
    --------------------------------------------------------------------
    Subtree, depth = 13 (5 data points).
    Split on feature home_ownership_RENT. (5, 0)
    --------------------------------------------------------------------
    Subtree, depth = 14 (5 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 13 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 8 (1353 data points).
    Split on feature grade_G. (1353, 0)
    --------------------------------------------------------------------
    Subtree, depth = 9 (1353 data points).
    Split on feature term_ 60 months. (0, 1353)
    --------------------------------------------------------------------
    Subtree, depth = 10 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 10 (1353 data points).
    Split on feature home_ownership_MORTGAGE. (710, 643)
    --------------------------------------------------------------------
    Subtree, depth = 11 (710 data points).
    Split on feature home_ownership_OWN. (602, 108)
    --------------------------------------------------------------------
    Subtree, depth = 12 (602 data points).
    Split on feature home_ownership_RENT. (0, 602)
    --------------------------------------------------------------------
    Subtree, depth = 13 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 13 (602 data points).
    Split on feature emp_length_1 year. (565, 37)
    --------------------------------------------------------------------
    Subtree, depth = 14 (565 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (37 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 12 (108 data points).
    Split on feature home_ownership_RENT. (108, 0)
    --------------------------------------------------------------------
    Subtree, depth = 13 (108 data points).
    Split on feature emp_length_1 year. (100, 8)
    --------------------------------------------------------------------
    Subtree, depth = 14 (100 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (8 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 13 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 11 (643 data points).
    Split on feature home_ownership_OWN. (643, 0)
    --------------------------------------------------------------------
    Subtree, depth = 12 (643 data points).
    Split on feature home_ownership_RENT. (643, 0)
    --------------------------------------------------------------------
    Subtree, depth = 13 (643 data points).
    Split on feature emp_length_1 year. (602, 41)
    --------------------------------------------------------------------
    Subtree, depth = 14 (602 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (41 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 13 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 12 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 9 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 7 (1 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 6 (2133 data points).
    Split on feature grade_F. (2133, 0)
    --------------------------------------------------------------------
    Subtree, depth = 7 (2133 data points).
    Split on feature grade_G. (2133, 0)
    --------------------------------------------------------------------
    Subtree, depth = 8 (2133 data points).
    Split on feature term_ 60 months. (0, 2133)
    --------------------------------------------------------------------
    Subtree, depth = 9 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 9 (2133 data points).
    Split on feature home_ownership_MORTGAGE. (1045, 1088)
    --------------------------------------------------------------------
    Subtree, depth = 10 (1045 data points).
    Split on feature home_ownership_OTHER. (1044, 1)
    --------------------------------------------------------------------
    Subtree, depth = 11 (1044 data points).
    Split on feature home_ownership_OWN. (879, 165)
    --------------------------------------------------------------------
    Subtree, depth = 12 (879 data points).
    Split on feature home_ownership_RENT. (0, 879)
    --------------------------------------------------------------------
    Subtree, depth = 13 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 13 (879 data points).
    Split on feature emp_length_1 year. (809, 70)
    --------------------------------------------------------------------
    Subtree, depth = 14 (809 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (70 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 12 (165 data points).
    Split on feature emp_length_9 years. (157, 8)
    --------------------------------------------------------------------
    Subtree, depth = 13 (157 data points).
    Split on feature home_ownership_RENT. (157, 0)
    --------------------------------------------------------------------
    Subtree, depth = 14 (157 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 13 (8 data points).
    Split on feature home_ownership_RENT. (8, 0)
    --------------------------------------------------------------------
    Subtree, depth = 14 (8 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 11 (1 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 10 (1088 data points).
    Split on feature home_ownership_OTHER. (1088, 0)
    --------------------------------------------------------------------
    Subtree, depth = 11 (1088 data points).
    Split on feature home_ownership_OWN. (1088, 0)
    --------------------------------------------------------------------
    Subtree, depth = 12 (1088 data points).
    Split on feature home_ownership_RENT. (1088, 0)
    --------------------------------------------------------------------
    Subtree, depth = 13 (1088 data points).
    Split on feature emp_length_1 year. (1035, 53)
    --------------------------------------------------------------------
    Subtree, depth = 14 (1035 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (53 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 13 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 12 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 11 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 8 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 7 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (2058 data points).
    Split on feature grade_E. (2058, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (2058 data points).
    Split on feature grade_F. (2058, 0)
    --------------------------------------------------------------------
    Subtree, depth = 7 (2058 data points).
    Split on feature grade_G. (2058, 0)
    --------------------------------------------------------------------
    Subtree, depth = 8 (2058 data points).
    Split on feature term_ 60 months. (0, 2058)
    --------------------------------------------------------------------
    Subtree, depth = 9 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 9 (2058 data points).
    Split on feature home_ownership_MORTGAGE. (923, 1135)
    --------------------------------------------------------------------
    Subtree, depth = 10 (923 data points).
    Split on feature home_ownership_OTHER. (922, 1)
    --------------------------------------------------------------------
    Subtree, depth = 11 (922 data points).
    Split on feature home_ownership_OWN. (762, 160)
    --------------------------------------------------------------------
    Subtree, depth = 12 (762 data points).
    Split on feature home_ownership_RENT. (0, 762)
    --------------------------------------------------------------------
    Subtree, depth = 13 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 13 (762 data points).
    Split on feature emp_length_1 year. (704, 58)
    --------------------------------------------------------------------
    Subtree, depth = 14 (704 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (58 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 12 (160 data points).
    Split on feature home_ownership_RENT. (160, 0)
    --------------------------------------------------------------------
    Subtree, depth = 13 (160 data points).
    Split on feature emp_length_1 year. (154, 6)
    --------------------------------------------------------------------
    Subtree, depth = 14 (154 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (6 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 13 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 11 (1 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 10 (1135 data points).
    Split on feature home_ownership_OTHER. (1135, 0)
    --------------------------------------------------------------------
    Subtree, depth = 11 (1135 data points).
    Split on feature home_ownership_OWN. (1135, 0)
    --------------------------------------------------------------------
    Subtree, depth = 12 (1135 data points).
    Split on feature home_ownership_RENT. (1135, 0)
    --------------------------------------------------------------------
    Subtree, depth = 13 (1135 data points).
    Split on feature emp_length_1 year. (1096, 39)
    --------------------------------------------------------------------
    Subtree, depth = 14 (1096 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (39 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 13 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 12 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 11 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 8 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 7 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 4 (2190 data points).
    Split on feature grade_D. (2190, 0)
    --------------------------------------------------------------------
    Subtree, depth = 5 (2190 data points).
    Split on feature grade_E. (2190, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (2190 data points).
    Split on feature grade_F. (2190, 0)
    --------------------------------------------------------------------
    Subtree, depth = 7 (2190 data points).
    Split on feature grade_G. (2190, 0)
    --------------------------------------------------------------------
    Subtree, depth = 8 (2190 data points).
    Split on feature term_ 60 months. (0, 2190)
    --------------------------------------------------------------------
    Subtree, depth = 9 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 9 (2190 data points).
    Split on feature home_ownership_MORTGAGE. (803, 1387)
    --------------------------------------------------------------------
    Subtree, depth = 10 (803 data points).
    Split on feature emp_length_4 years. (746, 57)
    --------------------------------------------------------------------
    Subtree, depth = 11 (746 data points).
    Split on feature home_ownership_OTHER. (746, 0)
    --------------------------------------------------------------------
    Subtree, depth = 12 (746 data points).
    Split on feature home_ownership_OWN. (598, 148)
    --------------------------------------------------------------------
    Subtree, depth = 13 (598 data points).
    Split on feature home_ownership_RENT. (0, 598)
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 14 (598 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 13 (148 data points).
    Split on feature emp_length_< 1 year. (137, 11)
    --------------------------------------------------------------------
    Subtree, depth = 14 (137 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (11 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 12 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 11 (57 data points).
    Split on feature home_ownership_OTHER. (57, 0)
    --------------------------------------------------------------------
    Subtree, depth = 12 (57 data points).
    Split on feature home_ownership_OWN. (49, 8)
    --------------------------------------------------------------------
    Subtree, depth = 13 (49 data points).
    Split on feature home_ownership_RENT. (0, 49)
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 14 (49 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 13 (8 data points).
    Split on feature home_ownership_RENT. (8, 0)
    --------------------------------------------------------------------
    Subtree, depth = 14 (8 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 12 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 10 (1387 data points).
    Split on feature emp_length_6 years. (1313, 74)
    --------------------------------------------------------------------
    Subtree, depth = 11 (1313 data points).
    Split on feature home_ownership_OTHER. (1313, 0)
    --------------------------------------------------------------------
    Subtree, depth = 12 (1313 data points).
    Split on feature home_ownership_OWN. (1313, 0)
    --------------------------------------------------------------------
    Subtree, depth = 13 (1313 data points).
    Split on feature home_ownership_RENT. (1313, 0)
    --------------------------------------------------------------------
    Subtree, depth = 14 (1313 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 13 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 12 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 11 (74 data points).
    Split on feature home_ownership_OTHER. (74, 0)
    --------------------------------------------------------------------
    Subtree, depth = 12 (74 data points).
    Split on feature home_ownership_OWN. (74, 0)
    --------------------------------------------------------------------
    Subtree, depth = 13 (74 data points).
    Split on feature home_ownership_RENT. (74, 0)
    --------------------------------------------------------------------
    Subtree, depth = 14 (74 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 13 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 12 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 8 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 7 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 3 (1048 data points).
    Split on feature emp_length_5 years. (969, 79)
    --------------------------------------------------------------------
    Subtree, depth = 4 (969 data points).
    Split on feature grade_C. (969, 0)
    --------------------------------------------------------------------
    Subtree, depth = 5 (969 data points).
    Split on feature grade_D. (969, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (969 data points).
    Split on feature grade_E. (969, 0)
    --------------------------------------------------------------------
    Subtree, depth = 7 (969 data points).
    Split on feature grade_F. (969, 0)
    --------------------------------------------------------------------
    Subtree, depth = 8 (969 data points).
    Split on feature grade_G. (969, 0)
    --------------------------------------------------------------------
    Subtree, depth = 9 (969 data points).
    Split on feature term_ 60 months. (0, 969)
    --------------------------------------------------------------------
    Subtree, depth = 10 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 10 (969 data points).
    Split on feature home_ownership_MORTGAGE. (367, 602)
    --------------------------------------------------------------------
    Subtree, depth = 11 (367 data points).
    Split on feature home_ownership_OTHER. (367, 0)
    --------------------------------------------------------------------
    Subtree, depth = 12 (367 data points).
    Split on feature home_ownership_OWN. (291, 76)
    --------------------------------------------------------------------
    Subtree, depth = 13 (291 data points).
    Split on feature home_ownership_RENT. (0, 291)
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 14 (291 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 13 (76 data points).
    Split on feature emp_length_9 years. (71, 5)
    --------------------------------------------------------------------
    Subtree, depth = 14 (71 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (5 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 12 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 11 (602 data points).
    Split on feature emp_length_9 years. (580, 22)
    --------------------------------------------------------------------
    Subtree, depth = 12 (580 data points).
    Split on feature emp_length_3 years. (545, 35)
    --------------------------------------------------------------------
    Subtree, depth = 13 (545 data points).
    Split on feature emp_length_4 years. (506, 39)
    --------------------------------------------------------------------
    Subtree, depth = 14 (506 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (39 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 13 (35 data points).
    Split on feature home_ownership_OTHER. (35, 0)
    --------------------------------------------------------------------
    Subtree, depth = 14 (35 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 12 (22 data points).
    Split on feature home_ownership_OTHER. (22, 0)
    --------------------------------------------------------------------
    Subtree, depth = 13 (22 data points).
    Split on feature home_ownership_OWN. (22, 0)
    --------------------------------------------------------------------
    Subtree, depth = 14 (22 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 13 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 9 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 8 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 7 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 4 (79 data points).
    Split on feature home_ownership_MORTGAGE. (34, 45)
    --------------------------------------------------------------------
    Subtree, depth = 5 (34 data points).
    Split on feature grade_C. (34, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (34 data points).
    Split on feature grade_D. (34, 0)
    --------------------------------------------------------------------
    Subtree, depth = 7 (34 data points).
    Split on feature grade_E. (34, 0)
    --------------------------------------------------------------------
    Subtree, depth = 8 (34 data points).
    Split on feature grade_F. (34, 0)
    --------------------------------------------------------------------
    Subtree, depth = 9 (34 data points).
    Split on feature grade_G. (34, 0)
    --------------------------------------------------------------------
    Subtree, depth = 10 (34 data points).
    Split on feature term_ 60 months. (0, 34)
    --------------------------------------------------------------------
    Subtree, depth = 11 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 11 (34 data points).
    Split on feature home_ownership_OTHER. (34, 0)
    --------------------------------------------------------------------
    Subtree, depth = 12 (34 data points).
    Split on feature home_ownership_OWN. (25, 9)
    --------------------------------------------------------------------
    Subtree, depth = 13 (25 data points).
    Split on feature home_ownership_RENT. (0, 25)
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 14 (25 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 13 (9 data points).
    Split on feature home_ownership_RENT. (9, 0)
    --------------------------------------------------------------------
    Subtree, depth = 14 (9 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 12 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 10 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 9 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 8 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 7 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (45 data points).
    Split on feature grade_C. (45, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (45 data points).
    Split on feature grade_D. (45, 0)
    --------------------------------------------------------------------
    Subtree, depth = 7 (45 data points).
    Split on feature grade_E. (45, 0)
    --------------------------------------------------------------------
    Subtree, depth = 8 (45 data points).
    Split on feature grade_F. (45, 0)
    --------------------------------------------------------------------
    Subtree, depth = 9 (45 data points).
    Split on feature grade_G. (45, 0)
    --------------------------------------------------------------------
    Subtree, depth = 10 (45 data points).
    Split on feature term_ 60 months. (0, 45)
    --------------------------------------------------------------------
    Subtree, depth = 11 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 11 (45 data points).
    Split on feature home_ownership_OTHER. (45, 0)
    --------------------------------------------------------------------
    Subtree, depth = 12 (45 data points).
    Split on feature home_ownership_OWN. (45, 0)
    --------------------------------------------------------------------
    Subtree, depth = 13 (45 data points).
    Split on feature home_ownership_RENT. (45, 0)
    --------------------------------------------------------------------
    Subtree, depth = 14 (45 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 13 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 12 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 10 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 9 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 8 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 7 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 2 (101 data points).
    Split on feature emp_length_n/a. (96, 5)
    --------------------------------------------------------------------
    Subtree, depth = 3 (96 data points).
    Split on feature emp_length_< 1 year. (85, 11)
    --------------------------------------------------------------------
    Subtree, depth = 4 (85 data points).
    Split on feature grade_B. (85, 0)
    --------------------------------------------------------------------
    Subtree, depth = 5 (85 data points).
    Split on feature grade_C. (85, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (85 data points).
    Split on feature grade_D. (85, 0)
    --------------------------------------------------------------------
    Subtree, depth = 7 (85 data points).
    Split on feature grade_E. (85, 0)
    --------------------------------------------------------------------
    Subtree, depth = 8 (85 data points).
    Split on feature grade_F. (85, 0)
    --------------------------------------------------------------------
    Subtree, depth = 9 (85 data points).
    Split on feature grade_G. (85, 0)
    --------------------------------------------------------------------
    Subtree, depth = 10 (85 data points).
    Split on feature term_ 60 months. (0, 85)
    --------------------------------------------------------------------
    Subtree, depth = 11 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 11 (85 data points).
    Split on feature home_ownership_MORTGAGE. (26, 59)
    --------------------------------------------------------------------
    Subtree, depth = 12 (26 data points).
    Split on feature emp_length_3 years. (24, 2)
    --------------------------------------------------------------------
    Subtree, depth = 13 (24 data points).
    Split on feature home_ownership_OTHER. (24, 0)
    --------------------------------------------------------------------
    Subtree, depth = 14 (24 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 13 (2 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 12 (59 data points).
    Split on feature home_ownership_OTHER. (59, 0)
    --------------------------------------------------------------------
    Subtree, depth = 13 (59 data points).
    Split on feature home_ownership_OWN. (59, 0)
    --------------------------------------------------------------------
    Subtree, depth = 14 (59 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 13 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 10 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 9 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 8 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 7 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 4 (11 data points).
    Split on feature grade_B. (11, 0)
    --------------------------------------------------------------------
    Subtree, depth = 5 (11 data points).
    Split on feature grade_C. (11, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (11 data points).
    Split on feature grade_D. (11, 0)
    --------------------------------------------------------------------
    Subtree, depth = 7 (11 data points).
    Split on feature grade_E. (11, 0)
    --------------------------------------------------------------------
    Subtree, depth = 8 (11 data points).
    Split on feature grade_F. (11, 0)
    --------------------------------------------------------------------
    Subtree, depth = 9 (11 data points).
    Split on feature grade_G. (11, 0)
    --------------------------------------------------------------------
    Subtree, depth = 10 (11 data points).
    Split on feature term_ 60 months. (0, 11)
    --------------------------------------------------------------------
    Subtree, depth = 11 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 11 (11 data points).
    Split on feature home_ownership_MORTGAGE. (8, 3)
    --------------------------------------------------------------------
    Subtree, depth = 12 (8 data points).
    Split on feature home_ownership_OTHER. (8, 0)
    --------------------------------------------------------------------
    Subtree, depth = 13 (8 data points).
    Split on feature home_ownership_OWN. (6, 2)
    --------------------------------------------------------------------
    Subtree, depth = 14 (6 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (2 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 13 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 12 (3 data points).
    Split on feature home_ownership_OTHER. (3, 0)
    --------------------------------------------------------------------
    Subtree, depth = 13 (3 data points).
    Split on feature home_ownership_OWN. (3, 0)
    --------------------------------------------------------------------
    Subtree, depth = 14 (3 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 13 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 10 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 9 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 8 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 7 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 3 (5 data points).
    Split on feature grade_B. (5, 0)
    --------------------------------------------------------------------
    Subtree, depth = 4 (5 data points).
    Split on feature grade_C. (5, 0)
    --------------------------------------------------------------------
    Subtree, depth = 5 (5 data points).
    Split on feature grade_D. (5, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (5 data points).
    Split on feature grade_E. (5, 0)
    --------------------------------------------------------------------
    Subtree, depth = 7 (5 data points).
    Split on feature grade_F. (5, 0)
    --------------------------------------------------------------------
    Subtree, depth = 8 (5 data points).
    Split on feature grade_G. (5, 0)
    --------------------------------------------------------------------
    Subtree, depth = 9 (5 data points).
    Split on feature term_ 60 months. (0, 5)
    --------------------------------------------------------------------
    Subtree, depth = 10 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 10 (5 data points).
    Split on feature home_ownership_MORTGAGE. (2, 3)
    --------------------------------------------------------------------
    Subtree, depth = 11 (2 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 11 (3 data points).
    Split on feature home_ownership_OTHER. (3, 0)
    --------------------------------------------------------------------
    Subtree, depth = 12 (3 data points).
    Split on feature home_ownership_OWN. (3, 0)
    --------------------------------------------------------------------
    Subtree, depth = 13 (3 data points).
    Split on feature home_ownership_RENT. (3, 0)
    --------------------------------------------------------------------
    Subtree, depth = 14 (3 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 13 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 12 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 9 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 8 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 7 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 4 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 1 (28001 data points).
    Split on feature grade_D. (23300, 4701)
    --------------------------------------------------------------------
    Subtree, depth = 2 (23300 data points).
    Split on feature grade_E. (22024, 1276)
    --------------------------------------------------------------------
    Subtree, depth = 3 (22024 data points).
    Split on feature grade_F. (21666, 358)
    --------------------------------------------------------------------
    Subtree, depth = 4 (21666 data points).
    Split on feature emp_length_n/a. (20734, 932)
    --------------------------------------------------------------------
    Subtree, depth = 5 (20734 data points).
    Split on feature grade_G. (20638, 96)
    --------------------------------------------------------------------
    Subtree, depth = 6 (20638 data points).
    Split on feature grade_A. (15839, 4799)
    --------------------------------------------------------------------
    Subtree, depth = 7 (15839 data points).
    Split on feature home_ownership_OTHER. (15811, 28)
    --------------------------------------------------------------------
    Subtree, depth = 8 (15811 data points).
    Split on feature grade_B. (6894, 8917)
    --------------------------------------------------------------------
    Subtree, depth = 9 (6894 data points).
    Split on feature home_ownership_MORTGAGE. (4102, 2792)
    --------------------------------------------------------------------
    Subtree, depth = 10 (4102 data points).
    Split on feature emp_length_4 years. (3768, 334)
    --------------------------------------------------------------------
    Subtree, depth = 11 (3768 data points).
    Split on feature emp_length_9 years. (3639, 129)
    --------------------------------------------------------------------
    Subtree, depth = 12 (3639 data points).
    Split on feature emp_length_2 years. (3123, 516)
    --------------------------------------------------------------------
    Subtree, depth = 13 (3123 data points).
    Split on feature grade_C. (0, 3123)
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 14 (3123 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 13 (516 data points).
    Split on feature home_ownership_OWN. (458, 58)
    --------------------------------------------------------------------
    Subtree, depth = 14 (458 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (58 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 12 (129 data points).
    Split on feature home_ownership_OWN. (113, 16)
    --------------------------------------------------------------------
    Subtree, depth = 13 (113 data points).
    Split on feature grade_C. (0, 113)
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 14 (113 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 13 (16 data points).
    Split on feature grade_C. (0, 16)
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 14 (16 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 11 (334 data points).
    Split on feature grade_C. (0, 334)
    --------------------------------------------------------------------
    Subtree, depth = 12 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 12 (334 data points).
    Split on feature term_ 60 months. (334, 0)
    --------------------------------------------------------------------
    Subtree, depth = 13 (334 data points).
    Split on feature home_ownership_OWN. (286, 48)
    --------------------------------------------------------------------
    Subtree, depth = 14 (286 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (48 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 13 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 10 (2792 data points).
    Split on feature emp_length_2 years. (2562, 230)
    --------------------------------------------------------------------
    Subtree, depth = 11 (2562 data points).
    Split on feature emp_length_5 years. (2335, 227)
    --------------------------------------------------------------------
    Subtree, depth = 12 (2335 data points).
    Split on feature grade_C. (0, 2335)
    --------------------------------------------------------------------
    Subtree, depth = 13 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 13 (2335 data points).
    Split on feature term_ 60 months. (2335, 0)
    --------------------------------------------------------------------
    Subtree, depth = 14 (2335 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 12 (227 data points).
    Split on feature grade_C. (0, 227)
    --------------------------------------------------------------------
    Subtree, depth = 13 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 13 (227 data points).
    Split on feature term_ 60 months. (227, 0)
    --------------------------------------------------------------------
    Subtree, depth = 14 (227 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 11 (230 data points).
    Split on feature grade_C. (0, 230)
    --------------------------------------------------------------------
    Subtree, depth = 12 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 12 (230 data points).
    Split on feature term_ 60 months. (230, 0)
    --------------------------------------------------------------------
    Subtree, depth = 13 (230 data points).
    Split on feature home_ownership_OWN. (230, 0)
    --------------------------------------------------------------------
    Subtree, depth = 14 (230 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 13 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 9 (8917 data points).
    Split on feature grade_C. (8917, 0)
    --------------------------------------------------------------------
    Subtree, depth = 10 (8917 data points).
    Split on feature term_ 60 months. (8917, 0)
    --------------------------------------------------------------------
    Subtree, depth = 11 (8917 data points).
    Split on feature home_ownership_MORTGAGE. (4748, 4169)
    --------------------------------------------------------------------
    Subtree, depth = 12 (4748 data points).
    Split on feature home_ownership_OWN. (4089, 659)
    --------------------------------------------------------------------
    Subtree, depth = 13 (4089 data points).
    Split on feature home_ownership_RENT. (0, 4089)
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 14 (4089 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 13 (659 data points).
    Split on feature home_ownership_RENT. (659, 0)
    --------------------------------------------------------------------
    Subtree, depth = 14 (659 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 12 (4169 data points).
    Split on feature home_ownership_OWN. (4169, 0)
    --------------------------------------------------------------------
    Subtree, depth = 13 (4169 data points).
    Split on feature home_ownership_RENT. (4169, 0)
    --------------------------------------------------------------------
    Subtree, depth = 14 (4169 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 13 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 11 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 10 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 8 (28 data points).
    Split on feature grade_B. (11, 17)
    --------------------------------------------------------------------
    Subtree, depth = 9 (11 data points).
    Split on feature emp_length_6 years. (10, 1)
    --------------------------------------------------------------------
    Subtree, depth = 10 (10 data points).
    Split on feature grade_C. (0, 10)
    --------------------------------------------------------------------
    Subtree, depth = 11 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 11 (10 data points).
    Split on feature term_ 60 months. (10, 0)
    --------------------------------------------------------------------
    Subtree, depth = 12 (10 data points).
    Split on feature home_ownership_MORTGAGE. (10, 0)
    --------------------------------------------------------------------
    Subtree, depth = 13 (10 data points).
    Split on feature home_ownership_OWN. (10, 0)
    --------------------------------------------------------------------
    Subtree, depth = 14 (10 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 13 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 12 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 10 (1 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 9 (17 data points).
    Split on feature emp_length_1 year. (16, 1)
    --------------------------------------------------------------------
    Subtree, depth = 10 (16 data points).
    Split on feature emp_length_3 years. (15, 1)
    --------------------------------------------------------------------
    Subtree, depth = 11 (15 data points).
    Split on feature emp_length_4 years. (14, 1)
    --------------------------------------------------------------------
    Subtree, depth = 12 (14 data points).
    Split on feature emp_length_< 1 year. (13, 1)
    --------------------------------------------------------------------
    Subtree, depth = 13 (13 data points).
    Split on feature grade_C. (13, 0)
    --------------------------------------------------------------------
    Subtree, depth = 14 (13 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 13 (1 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 12 (1 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 11 (1 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 10 (1 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 7 (4799 data points).
    Split on feature grade_B. (4799, 0)
    --------------------------------------------------------------------
    Subtree, depth = 8 (4799 data points).
    Split on feature grade_C. (4799, 0)
    --------------------------------------------------------------------
    Subtree, depth = 9 (4799 data points).
    Split on feature term_ 60 months. (4799, 0)
    --------------------------------------------------------------------
    Subtree, depth = 10 (4799 data points).
    Split on feature home_ownership_MORTGAGE. (2163, 2636)
    --------------------------------------------------------------------
    Subtree, depth = 11 (2163 data points).
    Split on feature home_ownership_OTHER. (2154, 9)
    --------------------------------------------------------------------
    Subtree, depth = 12 (2154 data points).
    Split on feature home_ownership_OWN. (1753, 401)
    --------------------------------------------------------------------
    Subtree, depth = 13 (1753 data points).
    Split on feature home_ownership_RENT. (0, 1753)
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 14 (1753 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 13 (401 data points).
    Split on feature home_ownership_RENT. (401, 0)
    --------------------------------------------------------------------
    Subtree, depth = 14 (401 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 12 (9 data points).
    Split on feature emp_length_3 years. (8, 1)
    --------------------------------------------------------------------
    Subtree, depth = 13 (8 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 13 (1 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 11 (2636 data points).
    Split on feature home_ownership_OTHER. (2636, 0)
    --------------------------------------------------------------------
    Subtree, depth = 12 (2636 data points).
    Split on feature home_ownership_OWN. (2636, 0)
    --------------------------------------------------------------------
    Subtree, depth = 13 (2636 data points).
    Split on feature home_ownership_RENT. (2636, 0)
    --------------------------------------------------------------------
    Subtree, depth = 14 (2636 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 13 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 12 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 10 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 9 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 8 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 6 (96 data points).
    Split on feature grade_A. (96, 0)
    --------------------------------------------------------------------
    Subtree, depth = 7 (96 data points).
    Split on feature grade_B. (96, 0)
    --------------------------------------------------------------------
    Subtree, depth = 8 (96 data points).
    Split on feature grade_C. (96, 0)
    --------------------------------------------------------------------
    Subtree, depth = 9 (96 data points).
    Split on feature term_ 60 months. (96, 0)
    --------------------------------------------------------------------
    Subtree, depth = 10 (96 data points).
    Split on feature home_ownership_MORTGAGE. (44, 52)
    --------------------------------------------------------------------
    Subtree, depth = 11 (44 data points).
    Split on feature emp_length_3 years. (43, 1)
    --------------------------------------------------------------------
    Subtree, depth = 12 (43 data points).
    Split on feature emp_length_7 years. (42, 1)
    --------------------------------------------------------------------
    Subtree, depth = 13 (42 data points).
    Split on feature emp_length_8 years. (41, 1)
    --------------------------------------------------------------------
    Subtree, depth = 14 (41 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (1 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 13 (1 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 12 (1 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 11 (52 data points).
    Split on feature emp_length_2 years. (47, 5)
    --------------------------------------------------------------------
    Subtree, depth = 12 (47 data points).
    Split on feature home_ownership_OTHER. (47, 0)
    --------------------------------------------------------------------
    Subtree, depth = 13 (47 data points).
    Split on feature home_ownership_OWN. (47, 0)
    --------------------------------------------------------------------
    Subtree, depth = 14 (47 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 13 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 12 (5 data points).
    Split on feature home_ownership_OTHER. (5, 0)
    --------------------------------------------------------------------
    Subtree, depth = 13 (5 data points).
    Split on feature home_ownership_OWN. (5, 0)
    --------------------------------------------------------------------
    Subtree, depth = 14 (5 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 13 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 10 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 9 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 8 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 7 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (932 data points).
    Split on feature grade_A. (702, 230)
    --------------------------------------------------------------------
    Subtree, depth = 6 (702 data points).
    Split on feature home_ownership_OTHER. (701, 1)
    --------------------------------------------------------------------
    Subtree, depth = 7 (701 data points).
    Split on feature grade_B. (317, 384)
    --------------------------------------------------------------------
    Subtree, depth = 8 (317 data points).
    Split on feature grade_C. (1, 316)
    --------------------------------------------------------------------
    Subtree, depth = 9 (1 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 9 (316 data points).
    Split on feature grade_G. (316, 0)
    --------------------------------------------------------------------
    Subtree, depth = 10 (316 data points).
    Split on feature term_ 60 months. (316, 0)
    --------------------------------------------------------------------
    Subtree, depth = 11 (316 data points).
    Split on feature home_ownership_MORTGAGE. (189, 127)
    --------------------------------------------------------------------
    Subtree, depth = 12 (189 data points).
    Split on feature home_ownership_OWN. (139, 50)
    --------------------------------------------------------------------
    Subtree, depth = 13 (139 data points).
    Split on feature home_ownership_RENT. (0, 139)
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 14 (139 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 13 (50 data points).
    Split on feature home_ownership_RENT. (50, 0)
    --------------------------------------------------------------------
    Subtree, depth = 14 (50 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 12 (127 data points).
    Split on feature home_ownership_OWN. (127, 0)
    --------------------------------------------------------------------
    Subtree, depth = 13 (127 data points).
    Split on feature home_ownership_RENT. (127, 0)
    --------------------------------------------------------------------
    Subtree, depth = 14 (127 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 13 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 11 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 10 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 8 (384 data points).
    Split on feature grade_C. (384, 0)
    --------------------------------------------------------------------
    Subtree, depth = 9 (384 data points).
    Split on feature grade_G. (384, 0)
    --------------------------------------------------------------------
    Subtree, depth = 10 (384 data points).
    Split on feature term_ 60 months. (384, 0)
    --------------------------------------------------------------------
    Subtree, depth = 11 (384 data points).
    Split on feature home_ownership_MORTGAGE. (210, 174)
    --------------------------------------------------------------------
    Subtree, depth = 12 (210 data points).
    Split on feature home_ownership_OWN. (148, 62)
    --------------------------------------------------------------------
    Subtree, depth = 13 (148 data points).
    Split on feature home_ownership_RENT. (0, 148)
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 14 (148 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 13 (62 data points).
    Split on feature home_ownership_RENT. (62, 0)
    --------------------------------------------------------------------
    Subtree, depth = 14 (62 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 12 (174 data points).
    Split on feature home_ownership_OWN. (174, 0)
    --------------------------------------------------------------------
    Subtree, depth = 13 (174 data points).
    Split on feature home_ownership_RENT. (174, 0)
    --------------------------------------------------------------------
    Subtree, depth = 14 (174 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 13 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 11 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 10 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 9 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 7 (1 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 6 (230 data points).
    Split on feature grade_B. (230, 0)
    --------------------------------------------------------------------
    Subtree, depth = 7 (230 data points).
    Split on feature grade_C. (230, 0)
    --------------------------------------------------------------------
    Subtree, depth = 8 (230 data points).
    Split on feature grade_G. (230, 0)
    --------------------------------------------------------------------
    Subtree, depth = 9 (230 data points).
    Split on feature term_ 60 months. (230, 0)
    --------------------------------------------------------------------
    Subtree, depth = 10 (230 data points).
    Split on feature home_ownership_MORTGAGE. (119, 111)
    --------------------------------------------------------------------
    Subtree, depth = 11 (119 data points).
    Split on feature home_ownership_OTHER. (119, 0)
    --------------------------------------------------------------------
    Subtree, depth = 12 (119 data points).
    Split on feature home_ownership_OWN. (71, 48)
    --------------------------------------------------------------------
    Subtree, depth = 13 (71 data points).
    Split on feature home_ownership_RENT. (0, 71)
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 14 (71 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 13 (48 data points).
    Split on feature home_ownership_RENT. (48, 0)
    --------------------------------------------------------------------
    Subtree, depth = 14 (48 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 12 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 11 (111 data points).
    Split on feature home_ownership_OTHER. (111, 0)
    --------------------------------------------------------------------
    Subtree, depth = 12 (111 data points).
    Split on feature home_ownership_OWN. (111, 0)
    --------------------------------------------------------------------
    Subtree, depth = 13 (111 data points).
    Split on feature home_ownership_RENT. (111, 0)
    --------------------------------------------------------------------
    Subtree, depth = 14 (111 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 13 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 12 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 10 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 9 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 8 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 7 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 4 (358 data points).
    Split on feature emp_length_8 years. (347, 11)
    --------------------------------------------------------------------
    Subtree, depth = 5 (347 data points).
    Split on feature grade_A. (347, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (347 data points).
    Split on feature grade_B. (347, 0)
    --------------------------------------------------------------------
    Subtree, depth = 7 (347 data points).
    Split on feature grade_C. (347, 0)
    --------------------------------------------------------------------
    Subtree, depth = 8 (347 data points).
    Split on feature grade_G. (347, 0)
    --------------------------------------------------------------------
    Subtree, depth = 9 (347 data points).
    Split on feature term_ 60 months. (347, 0)
    --------------------------------------------------------------------
    Subtree, depth = 10 (347 data points).
    Split on feature home_ownership_MORTGAGE. (237, 110)
    --------------------------------------------------------------------
    Subtree, depth = 11 (237 data points).
    Split on feature home_ownership_OTHER. (235, 2)
    --------------------------------------------------------------------
    Subtree, depth = 12 (235 data points).
    Split on feature home_ownership_OWN. (203, 32)
    --------------------------------------------------------------------
    Subtree, depth = 13 (203 data points).
    Split on feature home_ownership_RENT. (0, 203)
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 14 (203 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 13 (32 data points).
    Split on feature home_ownership_RENT. (32, 0)
    --------------------------------------------------------------------
    Subtree, depth = 14 (32 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 12 (2 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 11 (110 data points).
    Split on feature home_ownership_OTHER. (110, 0)
    --------------------------------------------------------------------
    Subtree, depth = 12 (110 data points).
    Split on feature home_ownership_OWN. (110, 0)
    --------------------------------------------------------------------
    Subtree, depth = 13 (110 data points).
    Split on feature home_ownership_RENT. (110, 0)
    --------------------------------------------------------------------
    Subtree, depth = 14 (110 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 13 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 12 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 10 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 9 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 8 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 7 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (11 data points).
    Split on feature home_ownership_OWN. (9, 2)
    --------------------------------------------------------------------
    Subtree, depth = 6 (9 data points).
    Split on feature grade_A. (9, 0)
    --------------------------------------------------------------------
    Subtree, depth = 7 (9 data points).
    Split on feature grade_B. (9, 0)
    --------------------------------------------------------------------
    Subtree, depth = 8 (9 data points).
    Split on feature grade_C. (9, 0)
    --------------------------------------------------------------------
    Subtree, depth = 9 (9 data points).
    Split on feature grade_G. (9, 0)
    --------------------------------------------------------------------
    Subtree, depth = 10 (9 data points).
    Split on feature term_ 60 months. (9, 0)
    --------------------------------------------------------------------
    Subtree, depth = 11 (9 data points).
    Split on feature home_ownership_MORTGAGE. (6, 3)
    --------------------------------------------------------------------
    Subtree, depth = 12 (6 data points).
    Split on feature home_ownership_OTHER. (6, 0)
    --------------------------------------------------------------------
    Subtree, depth = 13 (6 data points).
    Split on feature home_ownership_RENT. (0, 6)
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 14 (6 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 13 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 12 (3 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 11 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 10 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 9 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 8 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 7 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 6 (2 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 3 (1276 data points).
    Split on feature grade_A. (1276, 0)
    --------------------------------------------------------------------
    Subtree, depth = 4 (1276 data points).
    Split on feature grade_B. (1276, 0)
    --------------------------------------------------------------------
    Subtree, depth = 5 (1276 data points).
    Split on feature grade_C. (1276, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (1276 data points).
    Split on feature grade_F. (1276, 0)
    --------------------------------------------------------------------
    Subtree, depth = 7 (1276 data points).
    Split on feature grade_G. (1276, 0)
    --------------------------------------------------------------------
    Subtree, depth = 8 (1276 data points).
    Split on feature term_ 60 months. (1276, 0)
    --------------------------------------------------------------------
    Subtree, depth = 9 (1276 data points).
    Split on feature home_ownership_MORTGAGE. (855, 421)
    --------------------------------------------------------------------
    Subtree, depth = 10 (855 data points).
    Split on feature home_ownership_OTHER. (849, 6)
    --------------------------------------------------------------------
    Subtree, depth = 11 (849 data points).
    Split on feature home_ownership_OWN. (737, 112)
    --------------------------------------------------------------------
    Subtree, depth = 12 (737 data points).
    Split on feature home_ownership_RENT. (0, 737)
    --------------------------------------------------------------------
    Subtree, depth = 13 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 13 (737 data points).
    Split on feature emp_length_1 year. (670, 67)
    --------------------------------------------------------------------
    Subtree, depth = 14 (670 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (67 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 12 (112 data points).
    Split on feature home_ownership_RENT. (112, 0)
    --------------------------------------------------------------------
    Subtree, depth = 13 (112 data points).
    Split on feature emp_length_1 year. (102, 10)
    --------------------------------------------------------------------
    Subtree, depth = 14 (102 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (10 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 13 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 11 (6 data points).
    Split on feature home_ownership_OWN. (6, 0)
    --------------------------------------------------------------------
    Subtree, depth = 12 (6 data points).
    Split on feature home_ownership_RENT. (6, 0)
    --------------------------------------------------------------------
    Subtree, depth = 13 (6 data points).
    Split on feature emp_length_1 year. (6, 0)
    --------------------------------------------------------------------
    Subtree, depth = 14 (6 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 13 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 12 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 10 (421 data points).
    Split on feature emp_length_6 years. (408, 13)
    --------------------------------------------------------------------
    Subtree, depth = 11 (408 data points).
    Split on feature home_ownership_OTHER. (408, 0)
    --------------------------------------------------------------------
    Subtree, depth = 12 (408 data points).
    Split on feature home_ownership_OWN. (408, 0)
    --------------------------------------------------------------------
    Subtree, depth = 13 (408 data points).
    Split on feature home_ownership_RENT. (408, 0)
    --------------------------------------------------------------------
    Subtree, depth = 14 (408 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 13 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 12 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 11 (13 data points).
    Split on feature home_ownership_OTHER. (13, 0)
    --------------------------------------------------------------------
    Subtree, depth = 12 (13 data points).
    Split on feature home_ownership_OWN. (13, 0)
    --------------------------------------------------------------------
    Subtree, depth = 13 (13 data points).
    Split on feature home_ownership_RENT. (13, 0)
    --------------------------------------------------------------------
    Subtree, depth = 14 (13 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 13 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 12 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 9 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 8 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 7 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 4 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 2 (4701 data points).
    Split on feature grade_A. (4701, 0)
    --------------------------------------------------------------------
    Subtree, depth = 3 (4701 data points).
    Split on feature grade_B. (4701, 0)
    --------------------------------------------------------------------
    Subtree, depth = 4 (4701 data points).
    Split on feature grade_C. (4701, 0)
    --------------------------------------------------------------------
    Subtree, depth = 5 (4701 data points).
    Split on feature grade_E. (4701, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (4701 data points).
    Split on feature grade_F. (4701, 0)
    --------------------------------------------------------------------
    Subtree, depth = 7 (4701 data points).
    Split on feature grade_G. (4701, 0)
    --------------------------------------------------------------------
    Subtree, depth = 8 (4701 data points).
    Split on feature term_ 60 months. (4701, 0)
    --------------------------------------------------------------------
    Subtree, depth = 9 (4701 data points).
    Split on feature home_ownership_MORTGAGE. (3047, 1654)
    --------------------------------------------------------------------
    Subtree, depth = 10 (3047 data points).
    Split on feature home_ownership_OTHER. (3037, 10)
    --------------------------------------------------------------------
    Subtree, depth = 11 (3037 data points).
    Split on feature home_ownership_OWN. (2633, 404)
    --------------------------------------------------------------------
    Subtree, depth = 12 (2633 data points).
    Split on feature home_ownership_RENT. (0, 2633)
    --------------------------------------------------------------------
    Subtree, depth = 13 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 13 (2633 data points).
    Split on feature emp_length_1 year. (2392, 241)
    --------------------------------------------------------------------
    Subtree, depth = 14 (2392 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (241 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 12 (404 data points).
    Split on feature home_ownership_RENT. (404, 0)
    --------------------------------------------------------------------
    Subtree, depth = 13 (404 data points).
    Split on feature emp_length_1 year. (374, 30)
    --------------------------------------------------------------------
    Subtree, depth = 14 (374 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (30 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 13 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 11 (10 data points).
    Split on feature home_ownership_OWN. (10, 0)
    --------------------------------------------------------------------
    Subtree, depth = 12 (10 data points).
    Split on feature home_ownership_RENT. (10, 0)
    --------------------------------------------------------------------
    Subtree, depth = 13 (10 data points).
    Split on feature emp_length_1 year. (9, 1)
    --------------------------------------------------------------------
    Subtree, depth = 14 (9 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (1 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 13 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 12 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 10 (1654 data points).
    Split on feature emp_length_5 years. (1532, 122)
    --------------------------------------------------------------------
    Subtree, depth = 11 (1532 data points).
    Split on feature emp_length_3 years. (1414, 118)
    --------------------------------------------------------------------
    Subtree, depth = 12 (1414 data points).
    Split on feature emp_length_9 years. (1351, 63)
    --------------------------------------------------------------------
    Subtree, depth = 13 (1351 data points).
    Split on feature home_ownership_OTHER. (1351, 0)
    --------------------------------------------------------------------
    Subtree, depth = 14 (1351 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 13 (63 data points).
    Split on feature home_ownership_OTHER. (63, 0)
    --------------------------------------------------------------------
    Subtree, depth = 14 (63 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 12 (118 data points).
    Split on feature home_ownership_OTHER. (118, 0)
    --------------------------------------------------------------------
    Subtree, depth = 13 (118 data points).
    Split on feature home_ownership_OWN. (118, 0)
    --------------------------------------------------------------------
    Subtree, depth = 14 (118 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 13 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 11 (122 data points).
    Split on feature home_ownership_OTHER. (122, 0)
    --------------------------------------------------------------------
    Subtree, depth = 12 (122 data points).
    Split on feature home_ownership_OWN. (122, 0)
    --------------------------------------------------------------------
    Subtree, depth = 13 (122 data points).
    Split on feature home_ownership_RENT. (122, 0)
    --------------------------------------------------------------------
    Subtree, depth = 14 (122 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 13 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 12 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 9 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 8 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 7 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 4 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 3 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    

### Evaluating the models

Let us evaluate the models on the **train** and **validation** data. Let us start by evaluating the classification error on the training data:


```python
print "Training data, classification error (model 1):", evaluate_classification_error(model_1, train_data, target)
print "Training data, classification error (model 2):", evaluate_classification_error(model_2, train_data, target)
print "Training data, classification error (model 3):", evaluate_classification_error(model_3, train_data, target)
```

    Training data, classification error (model 1): 0.400037610144
    Training data, classification error (model 2): 0.381850419084
    Training data, classification error (model 3): 0.374462712229
    

Now evaluate the classification error on the validation data.


```python
print "Validation data, classification error (model 1):", evaluate_classification_error(model_1, validation_data, target)
print "Validation data, classification error (model 2):", evaluate_classification_error(model_2, validation_data, target)
print "Validation data, classification error (model 3):", evaluate_classification_error(model_3, validation_data, target)
```

    Validation data, classification error (model 1): 0.398104265403
    Validation data, classification error (model 2): 0.383778543731
    Validation data, classification error (model 3): 0.380008616975
    

**Quiz Question:** Which tree has the smallest error on the validation data? **model 3**

**Quiz Question:** Does the tree with the smallest error in the training data also have the smallest error in the validation data? **yes**

**Quiz Question:** Is it always true that the tree with the lowest classification error on the **training** set will result in the lowest classification error in the **validation** set? **no**


### Measuring the complexity of the tree

Recall in the lecture that we talked about deeper trees being more complex. We will measure the complexity of the tree as

```
  complexity(T) = number of leaves in the tree T
```

Here, we provide a function `count_leaves` that counts the number of leaves in a tree. Using this implementation, compute the number of nodes in `model_1`, `model_2`, and `model_3`. 


```python
def count_leaves(tree):
    if tree['is_leaf']:
        return 1
    return count_leaves(tree['left']) + count_leaves(tree['right'])
```

Compute the number of nodes in `model_1`, `model_2`, and `model_3`.


```python
print "number of leaves in model_1 is : {}".format(count_leaves(model_1))
print "number of leaves in model_2 is : {}".format(count_leaves(model_2))
print "number of leaves in model_3 is : {}".format(count_leaves(model_3))
```

    number of leaves in model_1 is : 4
    number of leaves in model_2 is : 41
    number of leaves in model_3 is : 341
    

**Quiz Question:** Which tree has the largest complexity? 

**model_3**

**Quiz Question:** Is it always true that the most complex tree will result in the lowest classification error in the **validation_set**?

**no**

# Exploring the effect of min_error

We will compare three models trained with different values of the stopping criterion. We intentionally picked models at the extreme ends (**negative**, **just right**, and **too positive**).

Train three models with these parameters:
1. **model_4**: `min_error_reduction = -1` (ignoring this early stopping condition)
2. **model_5**: `min_error_reduction = 0` (just right)
3. **model_6**: `min_error_reduction = 5` (too positive)

For each of these three, we set `max_depth = 6`, and `min_node_size = 0`.

** Note:** Each tree can take up to 30 seconds to train.


```python
model_4 = decision_tree_create(train_data, features, 'safe_loans', max_depth = 6, 
                                min_node_size = 0, min_error_reduction=-1)
model_5 = decision_tree_create(train_data, features, 'safe_loans', max_depth = 6, 
                                min_node_size = 0, min_error_reduction=0)
model_6 = decision_tree_create(train_data, features, 'safe_loans', max_depth = 6, 
                                min_node_size = 0, min_error_reduction=5)
```

    --------------------------------------------------------------------
    Subtree, depth = 0 (37224 data points).
    Split on feature term_ 36 months. (9223, 28001)
    --------------------------------------------------------------------
    Subtree, depth = 1 (9223 data points).
    Split on feature grade_A. (9122, 101)
    --------------------------------------------------------------------
    Subtree, depth = 2 (9122 data points).
    Split on feature grade_B. (8074, 1048)
    --------------------------------------------------------------------
    Subtree, depth = 3 (8074 data points).
    Split on feature grade_C. (5884, 2190)
    --------------------------------------------------------------------
    Subtree, depth = 4 (5884 data points).
    Split on feature grade_D. (3826, 2058)
    --------------------------------------------------------------------
    Subtree, depth = 5 (3826 data points).
    Split on feature grade_E. (1693, 2133)
    --------------------------------------------------------------------
    Subtree, depth = 6 (1693 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (2133 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 5 (2058 data points).
    Split on feature grade_E. (2058, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (2058 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 4 (2190 data points).
    Split on feature grade_D. (2190, 0)
    --------------------------------------------------------------------
    Subtree, depth = 5 (2190 data points).
    Split on feature grade_E. (2190, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (2190 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 3 (1048 data points).
    Split on feature emp_length_5 years. (969, 79)
    --------------------------------------------------------------------
    Subtree, depth = 4 (969 data points).
    Split on feature grade_C. (969, 0)
    --------------------------------------------------------------------
    Subtree, depth = 5 (969 data points).
    Split on feature grade_D. (969, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (969 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 4 (79 data points).
    Split on feature home_ownership_MORTGAGE. (34, 45)
    --------------------------------------------------------------------
    Subtree, depth = 5 (34 data points).
    Split on feature grade_C. (34, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (34 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (45 data points).
    Split on feature grade_C. (45, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (45 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 2 (101 data points).
    Split on feature emp_length_n/a. (96, 5)
    --------------------------------------------------------------------
    Subtree, depth = 3 (96 data points).
    Split on feature emp_length_< 1 year. (85, 11)
    --------------------------------------------------------------------
    Subtree, depth = 4 (85 data points).
    Split on feature grade_B. (85, 0)
    --------------------------------------------------------------------
    Subtree, depth = 5 (85 data points).
    Split on feature grade_C. (85, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (85 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 4 (11 data points).
    Split on feature grade_B. (11, 0)
    --------------------------------------------------------------------
    Subtree, depth = 5 (11 data points).
    Split on feature grade_C. (11, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (11 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 3 (5 data points).
    Split on feature grade_B. (5, 0)
    --------------------------------------------------------------------
    Subtree, depth = 4 (5 data points).
    Split on feature grade_C. (5, 0)
    --------------------------------------------------------------------
    Subtree, depth = 5 (5 data points).
    Split on feature grade_D. (5, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (5 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 4 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 1 (28001 data points).
    Split on feature grade_D. (23300, 4701)
    --------------------------------------------------------------------
    Subtree, depth = 2 (23300 data points).
    Split on feature grade_E. (22024, 1276)
    --------------------------------------------------------------------
    Subtree, depth = 3 (22024 data points).
    Split on feature grade_F. (21666, 358)
    --------------------------------------------------------------------
    Subtree, depth = 4 (21666 data points).
    Split on feature emp_length_n/a. (20734, 932)
    --------------------------------------------------------------------
    Subtree, depth = 5 (20734 data points).
    Split on feature grade_G. (20638, 96)
    --------------------------------------------------------------------
    Subtree, depth = 6 (20638 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (96 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 5 (932 data points).
    Split on feature grade_A. (702, 230)
    --------------------------------------------------------------------
    Subtree, depth = 6 (702 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (230 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 4 (358 data points).
    Split on feature emp_length_8 years. (347, 11)
    --------------------------------------------------------------------
    Subtree, depth = 5 (347 data points).
    Split on feature grade_A. (347, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (347 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (11 data points).
    Split on feature home_ownership_OWN. (9, 2)
    --------------------------------------------------------------------
    Subtree, depth = 6 (9 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (2 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 3 (1276 data points).
    Split on feature grade_A. (1276, 0)
    --------------------------------------------------------------------
    Subtree, depth = 4 (1276 data points).
    Split on feature grade_B. (1276, 0)
    --------------------------------------------------------------------
    Subtree, depth = 5 (1276 data points).
    Split on feature grade_C. (1276, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (1276 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 4 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 2 (4701 data points).
    Split on feature grade_A. (4701, 0)
    --------------------------------------------------------------------
    Subtree, depth = 3 (4701 data points).
    Split on feature grade_B. (4701, 0)
    --------------------------------------------------------------------
    Subtree, depth = 4 (4701 data points).
    Split on feature grade_C. (4701, 0)
    --------------------------------------------------------------------
    Subtree, depth = 5 (4701 data points).
    Split on feature grade_E. (4701, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (4701 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 4 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 3 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 0 (37224 data points).
    Split on feature term_ 36 months. (9223, 28001)
    --------------------------------------------------------------------
    Subtree, depth = 1 (9223 data points).
    Split on feature grade_A. (9122, 101)
    --------------------------------------------------------------------
    Subtree, depth = 2 (9122 data points).
    Early stopping condition 3 reached. Minimum error reduction.
    --------------------------------------------------------------------
    Subtree, depth = 2 (101 data points).
    Split on feature emp_length_n/a. (96, 5)
    --------------------------------------------------------------------
    Subtree, depth = 3 (96 data points).
    Split on feature emp_length_< 1 year. (85, 11)
    --------------------------------------------------------------------
    Subtree, depth = 4 (85 data points).
    Early stopping condition 3 reached. Minimum error reduction.
    --------------------------------------------------------------------
    Subtree, depth = 4 (11 data points).
    Early stopping condition 3 reached. Minimum error reduction.
    --------------------------------------------------------------------
    Subtree, depth = 3 (5 data points).
    Early stopping condition 3 reached. Minimum error reduction.
    --------------------------------------------------------------------
    Subtree, depth = 1 (28001 data points).
    Split on feature grade_D. (23300, 4701)
    --------------------------------------------------------------------
    Subtree, depth = 2 (23300 data points).
    Split on feature grade_E. (22024, 1276)
    --------------------------------------------------------------------
    Subtree, depth = 3 (22024 data points).
    Split on feature grade_F. (21666, 358)
    --------------------------------------------------------------------
    Subtree, depth = 4 (21666 data points).
    Split on feature emp_length_n/a. (20734, 932)
    --------------------------------------------------------------------
    Subtree, depth = 5 (20734 data points).
    Split on feature grade_G. (20638, 96)
    --------------------------------------------------------------------
    Subtree, depth = 6 (20638 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (96 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 5 (932 data points).
    Split on feature grade_A. (702, 230)
    --------------------------------------------------------------------
    Subtree, depth = 6 (702 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (230 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 4 (358 data points).
    Split on feature emp_length_8 years. (347, 11)
    --------------------------------------------------------------------
    Subtree, depth = 5 (347 data points).
    Early stopping condition 3 reached. Minimum error reduction.
    --------------------------------------------------------------------
    Subtree, depth = 5 (11 data points).
    Split on feature home_ownership_OWN. (9, 2)
    --------------------------------------------------------------------
    Subtree, depth = 6 (9 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (2 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 3 (1276 data points).
    Early stopping condition 3 reached. Minimum error reduction.
    --------------------------------------------------------------------
    Subtree, depth = 2 (4701 data points).
    Early stopping condition 3 reached. Minimum error reduction.
    --------------------------------------------------------------------
    Subtree, depth = 0 (37224 data points).
    Early stopping condition 3 reached. Minimum error reduction.
    

Calculate the accuracy of each model (**model_4**, **model_5**, or **model_6**) on the validation set. 


```python
print "Validation data, classification error (model 4):", evaluate_classification_error(model_4, validation_data, target)
print "Validation data, classification error (model 5):", evaluate_classification_error(model_5, validation_data, target)
print "Validation data, classification error (model 6):", evaluate_classification_error(model_6, validation_data, target)
```

    Validation data, classification error (model 4): 0.383778543731
    Validation data, classification error (model 5): 0.383778543731
    Validation data, classification error (model 6): 0.503446790177
    

Using the `count_leaves` function, compute the number of leaves in each of each models in (**model_4**, **model_5**, and **model_6**). 


```python
print "number of leaves in model_4 is : {}".format(count_leaves(model_4))
print "number of leaves in model_5 is : {}".format(count_leaves(model_5))
print "number of leaves in model_6 is : {}".format(count_leaves(model_6))
```

    number of leaves in model_4 is : 41
    number of leaves in model_5 is : 13
    number of leaves in model_6 is : 1
    

**Quiz Question:** Using the complexity definition above, which model (**model_4**, **model_5**, or **model_6**) has the largest complexity?

Did this match your expectation?

**model_4**

**Quiz Question:** **model_4** and **model_5** have similar classification error on the validation set but **model_5** has lower complexity. Should you pick **model_5** over **model_4**?

**model_5**


# Exploring the effect of min_node_size

We will compare three models trained with different values of the stopping criterion. Again, intentionally picked models at the extreme ends (**too small**, **just right**, and **just right**).

Train three models with these parameters:
1. **model_7**: min_node_size = 0 (too small)
2. **model_8**: min_node_size = 2000 (just right)
3. **model_9**: min_node_size = 50000 (too large)

For each of these three, we set `max_depth = 6`, and `min_error_reduction = -1`.

** Note:** Each tree can take up to 30 seconds to train.


```python
model_7 = decision_tree_create(train_data, features, 'safe_loans', max_depth = 6, 
                                min_node_size = 0, min_error_reduction=-1)
model_8 = decision_tree_create(train_data, features, 'safe_loans', max_depth = 6, 
                                min_node_size = 2000, min_error_reduction=-1)
model_9 = decision_tree_create(train_data, features, 'safe_loans', max_depth = 6, 
                                min_node_size = 50000, min_error_reduction=-1)
```

    --------------------------------------------------------------------
    Subtree, depth = 0 (37224 data points).
    Split on feature term_ 36 months. (9223, 28001)
    --------------------------------------------------------------------
    Subtree, depth = 1 (9223 data points).
    Split on feature grade_A. (9122, 101)
    --------------------------------------------------------------------
    Subtree, depth = 2 (9122 data points).
    Split on feature grade_B. (8074, 1048)
    --------------------------------------------------------------------
    Subtree, depth = 3 (8074 data points).
    Split on feature grade_C. (5884, 2190)
    --------------------------------------------------------------------
    Subtree, depth = 4 (5884 data points).
    Split on feature grade_D. (3826, 2058)
    --------------------------------------------------------------------
    Subtree, depth = 5 (3826 data points).
    Split on feature grade_E. (1693, 2133)
    --------------------------------------------------------------------
    Subtree, depth = 6 (1693 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (2133 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 5 (2058 data points).
    Split on feature grade_E. (2058, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (2058 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 4 (2190 data points).
    Split on feature grade_D. (2190, 0)
    --------------------------------------------------------------------
    Subtree, depth = 5 (2190 data points).
    Split on feature grade_E. (2190, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (2190 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 3 (1048 data points).
    Split on feature emp_length_5 years. (969, 79)
    --------------------------------------------------------------------
    Subtree, depth = 4 (969 data points).
    Split on feature grade_C. (969, 0)
    --------------------------------------------------------------------
    Subtree, depth = 5 (969 data points).
    Split on feature grade_D. (969, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (969 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 4 (79 data points).
    Split on feature home_ownership_MORTGAGE. (34, 45)
    --------------------------------------------------------------------
    Subtree, depth = 5 (34 data points).
    Split on feature grade_C. (34, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (34 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (45 data points).
    Split on feature grade_C. (45, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (45 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 2 (101 data points).
    Split on feature emp_length_n/a. (96, 5)
    --------------------------------------------------------------------
    Subtree, depth = 3 (96 data points).
    Split on feature emp_length_< 1 year. (85, 11)
    --------------------------------------------------------------------
    Subtree, depth = 4 (85 data points).
    Split on feature grade_B. (85, 0)
    --------------------------------------------------------------------
    Subtree, depth = 5 (85 data points).
    Split on feature grade_C. (85, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (85 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 4 (11 data points).
    Split on feature grade_B. (11, 0)
    --------------------------------------------------------------------
    Subtree, depth = 5 (11 data points).
    Split on feature grade_C. (11, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (11 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 3 (5 data points).
    Split on feature grade_B. (5, 0)
    --------------------------------------------------------------------
    Subtree, depth = 4 (5 data points).
    Split on feature grade_C. (5, 0)
    --------------------------------------------------------------------
    Subtree, depth = 5 (5 data points).
    Split on feature grade_D. (5, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (5 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 4 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 1 (28001 data points).
    Split on feature grade_D. (23300, 4701)
    --------------------------------------------------------------------
    Subtree, depth = 2 (23300 data points).
    Split on feature grade_E. (22024, 1276)
    --------------------------------------------------------------------
    Subtree, depth = 3 (22024 data points).
    Split on feature grade_F. (21666, 358)
    --------------------------------------------------------------------
    Subtree, depth = 4 (21666 data points).
    Split on feature emp_length_n/a. (20734, 932)
    --------------------------------------------------------------------
    Subtree, depth = 5 (20734 data points).
    Split on feature grade_G. (20638, 96)
    --------------------------------------------------------------------
    Subtree, depth = 6 (20638 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (96 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 5 (932 data points).
    Split on feature grade_A. (702, 230)
    --------------------------------------------------------------------
    Subtree, depth = 6 (702 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (230 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 4 (358 data points).
    Split on feature emp_length_8 years. (347, 11)
    --------------------------------------------------------------------
    Subtree, depth = 5 (347 data points).
    Split on feature grade_A. (347, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (347 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (11 data points).
    Split on feature home_ownership_OWN. (9, 2)
    --------------------------------------------------------------------
    Subtree, depth = 6 (9 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (2 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 3 (1276 data points).
    Split on feature grade_A. (1276, 0)
    --------------------------------------------------------------------
    Subtree, depth = 4 (1276 data points).
    Split on feature grade_B. (1276, 0)
    --------------------------------------------------------------------
    Subtree, depth = 5 (1276 data points).
    Split on feature grade_C. (1276, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (1276 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 4 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 2 (4701 data points).
    Split on feature grade_A. (4701, 0)
    --------------------------------------------------------------------
    Subtree, depth = 3 (4701 data points).
    Split on feature grade_B. (4701, 0)
    --------------------------------------------------------------------
    Subtree, depth = 4 (4701 data points).
    Split on feature grade_C. (4701, 0)
    --------------------------------------------------------------------
    Subtree, depth = 5 (4701 data points).
    Split on feature grade_E. (4701, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (4701 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 4 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 3 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 0 (37224 data points).
    Split on feature term_ 36 months. (9223, 28001)
    --------------------------------------------------------------------
    Subtree, depth = 1 (9223 data points).
    Split on feature grade_A. (9122, 101)
    --------------------------------------------------------------------
    Subtree, depth = 2 (9122 data points).
    Split on feature grade_B. (8074, 1048)
    --------------------------------------------------------------------
    Subtree, depth = 3 (8074 data points).
    Split on feature grade_C. (5884, 2190)
    --------------------------------------------------------------------
    Subtree, depth = 4 (5884 data points).
    Split on feature grade_D. (3826, 2058)
    --------------------------------------------------------------------
    Subtree, depth = 5 (3826 data points).
    Split on feature grade_E. (1693, 2133)
    --------------------------------------------------------------------
    Subtree, depth = 6 (1693 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (2133 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 5 (2058 data points).
    Split on feature grade_E. (2058, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (2058 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 4 (2190 data points).
    Split on feature grade_D. (2190, 0)
    --------------------------------------------------------------------
    Subtree, depth = 5 (2190 data points).
    Split on feature grade_E. (2190, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (2190 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 3 (1048 data points).
    Early stopping condition 2 reached. Reached minimum node size.
    --------------------------------------------------------------------
    Subtree, depth = 2 (101 data points).
    Early stopping condition 2 reached. Reached minimum node size.
    --------------------------------------------------------------------
    Subtree, depth = 1 (28001 data points).
    Split on feature grade_D. (23300, 4701)
    --------------------------------------------------------------------
    Subtree, depth = 2 (23300 data points).
    Split on feature grade_E. (22024, 1276)
    --------------------------------------------------------------------
    Subtree, depth = 3 (22024 data points).
    Split on feature grade_F. (21666, 358)
    --------------------------------------------------------------------
    Subtree, depth = 4 (21666 data points).
    Split on feature emp_length_n/a. (20734, 932)
    --------------------------------------------------------------------
    Subtree, depth = 5 (20734 data points).
    Split on feature grade_G. (20638, 96)
    --------------------------------------------------------------------
    Subtree, depth = 6 (20638 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (96 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 5 (932 data points).
    Early stopping condition 2 reached. Reached minimum node size.
    --------------------------------------------------------------------
    Subtree, depth = 4 (358 data points).
    Early stopping condition 2 reached. Reached minimum node size.
    --------------------------------------------------------------------
    Subtree, depth = 3 (1276 data points).
    Early stopping condition 2 reached. Reached minimum node size.
    --------------------------------------------------------------------
    Subtree, depth = 2 (4701 data points).
    Split on feature grade_A. (4701, 0)
    --------------------------------------------------------------------
    Subtree, depth = 3 (4701 data points).
    Split on feature grade_B. (4701, 0)
    --------------------------------------------------------------------
    Subtree, depth = 4 (4701 data points).
    Split on feature grade_C. (4701, 0)
    --------------------------------------------------------------------
    Subtree, depth = 5 (4701 data points).
    Split on feature grade_E. (4701, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (4701 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 4 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 3 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 0 (37224 data points).
    Early stopping condition 2 reached. Reached minimum node size.
    

Now, let us evaluate the models (**model_7**, **model_8**, or **model_9**) on the **validation_set**.


```python
print "Validation data, classification error (model 7):", evaluate_classification_error(model_7, validation_data, target)
print "Validation data, classification error (model 8):", evaluate_classification_error(model_8, validation_data, target)
print "Validation data, classification error (model 9):", evaluate_classification_error(model_9, validation_data, target)
```

    Validation data, classification error (model 7): 0.383778543731
    Validation data, classification error (model 8): 0.384532529082
    Validation data, classification error (model 9): 0.503446790177
    

Using the `count_leaves` function, compute the number of leaves in each of each models (**model_7**, **model_8**, and **model_9**). 


```python
print "number of leaves in model_7 is : {}".format(count_leaves(model_7))
print "number of leaves in model_8 is : {}".format(count_leaves(model_8))
print "number of leaves in model_9 is : {}".format(count_leaves(model_9))
```

    number of leaves in model_7 is : 41
    number of leaves in model_8 is : 19
    number of leaves in model_9 is : 1
    

**Quiz Question:** Using the results obtained in this section, which model (**model_7**, **model_8**, or **model_9**) would you choose to use?

**model_8**


```python

```
