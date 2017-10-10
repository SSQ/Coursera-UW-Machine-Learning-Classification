

```python
import numpy as np
import pandas as pd
import json
```

### 1. Load the dataset into a data frame named loans


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




```python
# safe_loans =  1 => safe
# safe_loans = -1 => risky
loans['safe_loans'] = loans['bad_loans'].apply(lambda x : +1 if x==0 else -1)

#loans = loans.remove_column('bad_loans')
loans = loans.drop('bad_loans', axis=1)
```


```python
features = ['grade',              # grade of the loan
            'term',               # the term of the loan
            'home_ownership',     # home_ownership status: own, mortgage or rent
            'emp_length',         # number of years of employment
           ]
target = 'safe_loans'
```


```python
loans = loans[features + [target]]
```


```python
loans.iloc[122602]
```




    grade                      E
    term               60 months
    home_ownership      MORTGAGE
    emp_length               n/a
    safe_loans                -1
    Name: 122602, dtype: object



## One-hot encoding


```python
categorical_variables = []
for feat_name, feat_type in zip(loans.columns, loans.dtypes):
    if feat_type == object:
        categorical_variables.append(feat_name)
        
for feature in categorical_variables:
    
    loans_one_hot_encoded = pd.get_dummies(loans[feature],prefix=feature)
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




```python
with open('module-5-assignment-2-train-idx.json') as train_data_file:    
    train_idx  = json.load(train_data_file)
with open('module-5-assignment-2-test-idx.json') as test_data_file:    
    test_idx = json.load(test_data_file)

print train_idx[:3]
print test_idx[:3]
```

    [1, 6, 7]
    [24, 41, 60]
    


```python
print len(train_idx)
print len(test_idx)
```

    37224
    9284
    


```python
train_data = loans.iloc[train_idx]
test_data = loans.iloc[test_idx]
```


```python
print len(loans.dtypes )
```

    26
    

## Decision tree implementation

## Function to count number of mistakes while predicting majority class
Recall from the lecture that prediction at an intermediate node works by predicting the majority class for all data points that belong to this node. Now, we will write a function that calculates the number of misclassified examples when predicting the majority class. This will be used to help determine which feature is the best to split on at a given node of the tree.

Note: Keep in mind that in order to compute the number of mistakes for a majority classifier, we only need the label (y values) of the data points in the node.

Steps to follow:

- Step 1: Calculate the number of safe loans and risky loans.
- Step 2: Since we are assuming majority class prediction, all the data points that are not in the majority class are considered mistakes.
- Step 3: Return the number of mistakes.

### 7. Now, let us write the function intermediate_node_num_mistakes which computes the number of misclassified examples of an intermediate node given the set of labels (y values) of the data points contained in the node. Your code should be analogous to 


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

### 8. Because there are several steps in this assignment, we have introduced some stopping points where you can check your code and make sure it is correct before proceeding. To test your intermediate_node_num_mistakes function, run the following code until you get a Test passed!, then you should proceed. Otherwise, you should spend some time figuring out where things went wrong. Again, remember that this code is specific to SFrame, but using your software of choice, you can construct similar tests.


```python
# Test case 1
example_labels = np.array([-1, -1, 1, 1, 1])
if intermediate_node_num_mistakes(example_labels) == 2:
    print 'Test passed!'
else:
    print 'Test 1 failed... try again!'

# Test case 2
example_labels = np.array([-1, -1, 1, 1, 1, 1, 1])
if intermediate_node_num_mistakes(example_labels) == 2:
    print 'Test passed!'
else:
    print 'Test 3 failed... try again!'
    
# Test case 3
example_labels = np.array([-1, -1, -1, -1, -1, 1, 1])
if intermediate_node_num_mistakes(example_labels) == 2:
    print 'Test passed!'
else:
    print 'Test 3 failed... try again!'
```

    Test passed!
    Test passed!
    Test passed!
    

## Function to pick best feature to split on

The function best_splitting_feature takes 3 arguments:

- The data
- The features to consider for splits (a list of strings of column names to consider for splits)
- The name of the target/label column (string)

The function will loop through the list of possible features, and consider splitting on each of them. It will calculate the classification error of each split and return the feature that had the smallest classification error when split on.

Recall that the classification error is defined as follows:

### 9. Follow these steps to implement best_splitting_feature:

- Step 1: Loop over each feature in the feature list
- Step 2: Within the loop, split the data into two groups: one group where all of the data has feature value 0 or False (we will call this the left split), and one group where all of the data has feature value 1 or True (we will call this the right split). Make sure the left split corresponds with 0 and the right split corresponds with 1 to ensure your implementation fits with our implementation of the tree building process.
- Step 3: Calculate the number of misclassified examples in both groups of data and use the above formula to compute theclassification error.
- Step 4: If the computed error is smaller than the best error found so far, store this feature and its error.

Note: Remember that since we are only dealing with binary features, we do not have to consider thresholds for real-valued features. This makes the implementation of this function much easier.

Your code should be analogous to


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

## Building the tree

With the above functions implemented correctly, we are now ready to build our decision tree. Each node in the decision tree is represented as a dictionary which contains the following keys and possible values:

### 10. First, we will write a function that creates a leaf node given a set of target values. 
Your code should be analogous to


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

## 11. Now, we will provide a Python skeleton of the learning algorithm. Note that this code is not complete; it needs to be completed by you if you are using Python. Otherwise, your code should be analogous to
1. Stopping condition 1: All data points in a node are from the same class.
1. Stopping condition 2: No more features to split on.
1. Additional stopping condition: In addition to the above two stopping conditions covered in lecture, in this assignment we will also consider a stopping condition based on the max_depth of the tree. By not letting the tree grow too deep, we will save computational effort in the learning process.





```python
def decision_tree_create(data, features, target, current_depth = 0, max_depth = 10):
    remaining_features = features[:] # Make a copy of the features.
    
    target_values = data[target]
    print "--------------------------------------------------------------------"
    print "Subtree, depth = %s (%s data points)." % (current_depth, len(target_values))
    

    # Stopping condition 1
    # (Check if there are mistakes at current node.
    # Recall you wrote a function intermediate_node_num_mistakes to compute this.)
    if intermediate_node_num_mistakes(target_values) == 0:  ## YOUR CODE HERE
        print "Stopping condition 1 reached."     
        # If not mistakes at current node, make current node a leaf node
        return create_leaf(target_values)
    
    # Stopping condition 2 (check if there are remaining features to consider splitting on)
    if remaining_features == []:   ## YOUR CODE HERE
        print "Stopping condition 2 reached."    
        # If there are no remaining features to consider, make current node a leaf node
        return create_leaf(target_values)    
    
    # Additional stopping condition (limit tree depth)
    if current_depth >= max_depth:  ## YOUR CODE HERE
        print "Reached maximum depth. Stopping for now."
        # If the max tree depth has been reached, make current node a leaf node
        return create_leaf(target_values)

    # Find the best splitting feature (recall the function best_splitting_feature implemented above)
    ## YOUR CODE HERE
    splitting_feature = best_splitting_feature(data, remaining_features, target)
    
    # Split on the best feature that we found. 
    left_split = data[data[splitting_feature] == 0]
    right_split = data[data[splitting_feature] == 1]      ## YOUR CODE HERE
    remaining_features.remove(splitting_feature)
    print "Split on feature %s. (%s, %s)" % (\
                      splitting_feature, len(left_split), len(right_split))
    
    # Create a leaf node if the split is "perfect"
    if len(left_split) == len(data):
        print "Creating leaf node."
        return create_leaf(left_split[target])
    if len(right_split) == len(data):
        print "Creating leaf node."
        ## YOUR CODE HERE
        return create_leaf(right_split[target])
        
    # Repeat (recurse) on left and right subtrees
    left_tree = decision_tree_create(left_split, remaining_features, target, current_depth + 1, max_depth)        
    ## YOUR CODE HERE
    right_tree = decision_tree_create(right_split, remaining_features, target, current_depth + 1, max_depth)

    return {'is_leaf'          : False, 
            'prediction'       : None,
            'splitting_feature': splitting_feature,
            'left'             : left_tree, 
            'right'            : right_tree}
```

### 12. Train a tree model on the train_data. Limit the depth to 6 (max_depth = 6) to make sure the algorithm doesn't run for too long. Call this tree my_decision_tree. Warning: The tree may take 1-2 minutes to learn.


```python
input_features = train_data.columns
print list(input_features)
```

    ['safe_loans', 'grade_A', 'grade_B', 'grade_C', 'grade_D', 'grade_E', 'grade_F', 'grade_G', 'term_ 36 months', 'term_ 60 months', 'home_ownership_MORTGAGE', 'home_ownership_OTHER', 'home_ownership_OWN', 'home_ownership_RENT', 'emp_length_1 year', 'emp_length_10+ years', 'emp_length_2 years', 'emp_length_3 years', 'emp_length_4 years', 'emp_length_5 years', 'emp_length_6 years', 'emp_length_7 years', 'emp_length_8 years', 'emp_length_9 years', 'emp_length_< 1 year', 'emp_length_n/a']
    


```python
a = list(train_data.columns)
a.remove('safe_loans')
print a
print list(train_data.columns)
```

    ['grade_A', 'grade_B', 'grade_C', 'grade_D', 'grade_E', 'grade_F', 'grade_G', 'term_ 36 months', 'term_ 60 months', 'home_ownership_MORTGAGE', 'home_ownership_OTHER', 'home_ownership_OWN', 'home_ownership_RENT', 'emp_length_1 year', 'emp_length_10+ years', 'emp_length_2 years', 'emp_length_3 years', 'emp_length_4 years', 'emp_length_5 years', 'emp_length_6 years', 'emp_length_7 years', 'emp_length_8 years', 'emp_length_9 years', 'emp_length_< 1 year', 'emp_length_n/a']
    ['safe_loans', 'grade_A', 'grade_B', 'grade_C', 'grade_D', 'grade_E', 'grade_F', 'grade_G', 'term_ 36 months', 'term_ 60 months', 'home_ownership_MORTGAGE', 'home_ownership_OTHER', 'home_ownership_OWN', 'home_ownership_RENT', 'emp_length_1 year', 'emp_length_10+ years', 'emp_length_2 years', 'emp_length_3 years', 'emp_length_4 years', 'emp_length_5 years', 'emp_length_6 years', 'emp_length_7 years', 'emp_length_8 years', 'emp_length_9 years', 'emp_length_< 1 year', 'emp_length_n/a']
    


```python
my_decision_tree = decision_tree_create(train_data, a, 'safe_loans', current_depth = 0, max_depth = 6)
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
    Reached maximum depth. Stopping for now.
    --------------------------------------------------------------------
    Subtree, depth = 6 (2133 data points).
    Reached maximum depth. Stopping for now.
    --------------------------------------------------------------------
    Subtree, depth = 5 (2058 data points).
    Split on feature grade_E. (2058, 0)
    Creating leaf node.
    --------------------------------------------------------------------
    Subtree, depth = 4 (2190 data points).
    Split on feature grade_D. (2190, 0)
    Creating leaf node.
    --------------------------------------------------------------------
    Subtree, depth = 3 (1048 data points).
    Split on feature emp_length_5 years. (969, 79)
    --------------------------------------------------------------------
    Subtree, depth = 4 (969 data points).
    Split on feature grade_C. (969, 0)
    Creating leaf node.
    --------------------------------------------------------------------
    Subtree, depth = 4 (79 data points).
    Split on feature home_ownership_MORTGAGE. (34, 45)
    --------------------------------------------------------------------
    Subtree, depth = 5 (34 data points).
    Split on feature grade_C. (34, 0)
    Creating leaf node.
    --------------------------------------------------------------------
    Subtree, depth = 5 (45 data points).
    Split on feature grade_C. (45, 0)
    Creating leaf node.
    --------------------------------------------------------------------
    Subtree, depth = 2 (101 data points).
    Split on feature emp_length_n/a. (96, 5)
    --------------------------------------------------------------------
    Subtree, depth = 3 (96 data points).
    Split on feature emp_length_< 1 year. (85, 11)
    --------------------------------------------------------------------
    Subtree, depth = 4 (85 data points).
    Split on feature grade_B. (85, 0)
    Creating leaf node.
    --------------------------------------------------------------------
    Subtree, depth = 4 (11 data points).
    Split on feature grade_B. (11, 0)
    Creating leaf node.
    --------------------------------------------------------------------
    Subtree, depth = 3 (5 data points).
    Split on feature grade_B. (5, 0)
    Creating leaf node.
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
    Reached maximum depth. Stopping for now.
    --------------------------------------------------------------------
    Subtree, depth = 6 (96 data points).
    Reached maximum depth. Stopping for now.
    --------------------------------------------------------------------
    Subtree, depth = 5 (932 data points).
    Split on feature grade_A. (702, 230)
    --------------------------------------------------------------------
    Subtree, depth = 6 (702 data points).
    Reached maximum depth. Stopping for now.
    --------------------------------------------------------------------
    Subtree, depth = 6 (230 data points).
    Reached maximum depth. Stopping for now.
    --------------------------------------------------------------------
    Subtree, depth = 4 (358 data points).
    Split on feature emp_length_8 years. (347, 11)
    --------------------------------------------------------------------
    Subtree, depth = 5 (347 data points).
    Split on feature grade_A. (347, 0)
    Creating leaf node.
    --------------------------------------------------------------------
    Subtree, depth = 5 (11 data points).
    Split on feature home_ownership_OWN. (9, 2)
    --------------------------------------------------------------------
    Subtree, depth = 6 (9 data points).
    Reached maximum depth. Stopping for now.
    --------------------------------------------------------------------
    Subtree, depth = 6 (2 data points).
    Stopping condition 1 reached.
    --------------------------------------------------------------------
    Subtree, depth = 3 (1276 data points).
    Split on feature grade_A. (1276, 0)
    Creating leaf node.
    --------------------------------------------------------------------
    Subtree, depth = 2 (4701 data points).
    Split on feature grade_A. (4701, 0)
    Creating leaf node.
    

## Making predictions with a decision tree

### 13. As discussed in the lecture, we can make predictions from the decision tree with a simple recursive function. Write a function called classify, which takes in a learned tree and a test point x to classify. Include an option annotate that describes the prediction path when set to True. Your code should be analogous to


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

### 14. Now, let's consider the first example of the test set and see what my_decision_tree model predicts for this data point.


```python
print test_data.iloc[0]
print 'Predicted class: %s ' % classify(my_decision_tree, test_data.iloc[0])
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
    Predicted class: -1 
    

### 15. Let's add some annotations to our prediction to see what the prediction path was that lead to this predicted class:


```python
classify(my_decision_tree, test_data.iloc[0], annotate=True)
```

    Split on term_ 36 months = 0
    Split on grade_A = 0
    Split on grade_B = 0
    Split on grade_C = 0
    Split on grade_D = 1
    At leaf, predicting -1
    




    -1



## Quiz question: 
What was the feature that my_decision_tree first split on while making the prediction for test_data[0]?

## Quiz question: 
What was the first feature that lead to a right split of test_data[0]?

## Quiz question:
What was the last feature split on before reaching a leaf node for test_data[0]?

## Answer: 
term_36 months
## Answer: 
grade_D
## Answer: 
grade_D

## Evaluating your decision tree

### 16. Now, we will write a function to evaluate a decision tree by computing the classification error of the tree on the given dataset. Write a function called evaluate_classification_error that takes in as input:

- tree (as described above)
- data (a data frame of data points)

This function should return a prediction (class label) for each row in data using the decision tree. Your code should be analogous to


```python
def evaluate_classification_error(tree, data):
    # Apply the classify(tree, x) to each row in your data
    prediction = data.apply(lambda x: classify(tree, x), axis=1)
    
    # Once you've made the predictions, calculate the classification error and return it
    ## YOUR CODE HERE
    
    return (data['safe_loans'] != np.array(prediction)).values.sum() *1. / len(data)
```

### 17. Now, use this function to evaluate the classification error on the test set.


```python
evaluate_classification_error(my_decision_tree, test_data)
```




    0.38377854373115039




```python
print 1-0.38
```

    0.62
    

## Quiz Question: 
Rounded to 2nd decimal point, what is the classification error of my_decision_tree on the test_data?

## Answer:
0.38

## Printing out a decision stump

### 18. As discussed in the lecture, we can print out a single decision stump (printing out the entire tree is left as an exercise to the curious reader). Here we provide Python code to visualize a decision stump. If you are using different software, make sure your code is analogous to:




```python
def print_stump(tree, name = 'root'):
    split_name = tree['splitting_feature'] # split_name is something like 'term. 36 months'
    if split_name is None:
        print "(leaf, label: %s)" % tree['prediction']
        return None
    split_feature, split_value = split_name.split('_',1)
    print '                       %s' % name
    print '         |---------------|----------------|'
    print '         |                                |'
    print '         |                                |'
    print '         |                                |'
    print '  [{0} == 0]               [{0} == 1]    '.format(split_name)
    print '         |                                |'
    print '         |                                |'
    print '         |                                |'
    print '    (%s)                         (%s)' \
        % (('leaf, label: ' + str(tree['left']['prediction']) if tree['left']['is_leaf'] else 'subtree'),
           ('leaf, label: ' + str(tree['right']['prediction']) if tree['right']['is_leaf'] else 'subtree'))
```

### 19. Using this function, we can print out the root of our decision tree:


```python
print_stump(my_decision_tree)
```

                           root
             |---------------|----------------|
             |                                |
             |                                |
             |                                |
      [term_ 36 months == 0]               [term_ 36 months == 1]    
             |                                |
             |                                |
             |                                |
        (subtree)                         (subtree)
    

## Quiz Question: 
What is the feature that is used for the split at the root node?

## Answer:
term_ 36 months

## Exploring the intermediate left subtree
The tree is a recursive dictionary, so we do have access to all the nodes! We can use

- my_decision_tree['left'] to go left
- my_decision_tree['right'] to go right

### 20. We can print out the left subtree by running the code


```python
print_stump(my_decision_tree['left'], my_decision_tree['splitting_feature'])
```

                           term_ 36 months
             |---------------|----------------|
             |                                |
             |                                |
             |                                |
      [grade_A == 0]               [grade_A == 1]    
             |                                |
             |                                |
             |                                |
        (subtree)                         (subtree)
    


```python
print_stump(my_decision_tree['left']['left'], my_decision_tree['left']['splitting_feature'])
```

                           grade_A
             |---------------|----------------|
             |                                |
             |                                |
             |                                |
      [grade_B == 0]               [grade_B == 1]    
             |                                |
             |                                |
             |                                |
        (subtree)                         (subtree)
    


```python
print_stump(my_decision_tree['right'], my_decision_tree['splitting_feature'])
```

                           term_ 36 months
             |---------------|----------------|
             |                                |
             |                                |
             |                                |
      [grade_D == 0]               [grade_D == 1]    
             |                                |
             |                                |
             |                                |
        (subtree)                         (leaf, label: -1)
    


```python
print_stump(my_decision_tree['right']['right'], my_decision_tree['right']['splitting_feature'])
```

    (leaf, label: -1)
    

## Quiz question: 
What is the path of the first 3 feature splits considered along the left-most branch of my_decision_tree?

## Quiz question: 
What is the path of the first 3 feature splits considered along the right-most branch of my_decision_tree?

## Answer
- term_ 36 months
- grade_A
- grade_B

## Answer
- term_ 36 months
- grade_D
- leaf


```python

```
