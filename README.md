

```python
# Reload modules before executing user code
%reload_ext autoreload
# Reload all modules (except those excluded by %aimport)
%autoreload 2
# Show plots within this notebook
%matplotlib inline
```

# 1. Load our training and test data into pandas dataframes


```python
PATH='download/'
test_csv = f'{PATH}test.csv'
train_csv = f'{PATH}train.csv'
sample_submission_csv = f'{PATH}sample_submission.csv'
```


```python
import pandas as pd

train_dataframe = pd.read_csv(train_csv, na_filter=False)
test_dataframe = pd.read_csv(test_csv, na_filter=False)
```

# 2. View data

The labels are all in the same scale and won't need to be standardized.


```python
train_dataframe.loc[train_dataframe['threat'] == 1].head(1)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>comment_text</th>
      <th>toxic</th>
      <th>severe_toxic</th>
      <th>obscene</th>
      <th>threat</th>
      <th>insult</th>
      <th>identity_hate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>79</th>
      <td>003217c3eb469ba9</td>
      <td>Hi! I am back again!\nLast warning!\nStop undo...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_dataframe.shape
```




    (159571, 8)




```python
train_dataframe.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>toxic</th>
      <th>severe_toxic</th>
      <th>obscene</th>
      <th>threat</th>
      <th>insult</th>
      <th>identity_hate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>159571.000000</td>
      <td>159571.000000</td>
      <td>159571.000000</td>
      <td>159571.000000</td>
      <td>159571.000000</td>
      <td>159571.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.095844</td>
      <td>0.009996</td>
      <td>0.052948</td>
      <td>0.002996</td>
      <td>0.049364</td>
      <td>0.008805</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.294379</td>
      <td>0.099477</td>
      <td>0.223931</td>
      <td>0.054650</td>
      <td>0.216627</td>
      <td>0.093420</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_dataframe.info() # verify that are no missing values in our dataset
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 159571 entries, 0 to 159570
    Data columns (total 8 columns):
    id               159571 non-null object
    comment_text     159571 non-null object
    toxic            159571 non-null int64
    severe_toxic     159571 non-null int64
    obscene          159571 non-null int64
    threat           159571 non-null int64
    insult           159571 non-null int64
    identity_hate    159571 non-null int64
    dtypes: int64(6), object(2)
    memory usage: 9.7+ MB


# 4. Separate target features (y) from input features (X) 


Use sklearn.model_selection.train_test_split to split training data into validation and train. 


```python
from sklearn.model_selection import train_test_split 

X = train_dataframe['comment_text']
y = train_dataframe[['obscene','insult','toxic','severe_toxic','identity_hate','threat']]

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=1)
X_test = test_dataframe['comment_text']
```

# 5. Tokenize words from comments


```python
from sklearn.feature_extraction.text import TfidfVectorizer

# create the transform
vectorizer = TfidfVectorizer()

# tokenize and build vocab with training data
X_train_tokenized = vectorizer.fit_transform(X_train)

# transform validation and test data to have the same shape
X_valid_tokenized = vectorizer.transform(X_valid)
X_test_tokenized = vectorizer.transform(X_test)
```


```python
# examine the vocabulary and document-term matrix together
pd.DataFrame(X_train_tokenized.toarray(), columns=vectorizer.get_feature_names()).head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>00</th>
      <th>000</th>
      <th>0000</th>
      <th>00000</th>
      <th>000000</th>
      <th>0000000</th>
      <th>0000000027</th>
      <th>00000001</th>
      <th>00000003</th>
      <th>00000050</th>
      <th>...</th>
      <th>천리마군</th>
      <th>칠지도</th>
      <th>ﬂute</th>
      <th>ａｎｏｎｔａｌｋ</th>
      <th>ｃｏｍ</th>
      <th>ｗｗｗ</th>
      <th>ｳｨｷﾍﾟﾃﾞｨｱ</th>
      <th>𐌰𐌹</th>
      <th>𐌰𐌿</th>
      <th>𐌴𐌹</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 165609 columns</p>
</div>




```python
pd.DataFrame(X_test_tokenized.toarray(), columns=vectorizer.get_feature_names()).head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>00</th>
      <th>000</th>
      <th>0000</th>
      <th>00000</th>
      <th>000000</th>
      <th>0000000</th>
      <th>0000000027</th>
      <th>00000001</th>
      <th>00000003</th>
      <th>00000050</th>
      <th>...</th>
      <th>천리마군</th>
      <th>칠지도</th>
      <th>ﬂute</th>
      <th>ａｎｏｎｔａｌｋ</th>
      <th>ｃｏｍ</th>
      <th>ｗｗｗ</th>
      <th>ｳｨｷﾍﾟﾃﾞｨｱ</th>
      <th>𐌰𐌹</th>
      <th>𐌰𐌿</th>
      <th>𐌴𐌹</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 165609 columns</p>
</div>



# 6. Problem transformation

Train one binary classifier for each label. This is called binary relevance. 


```python
# Prepare submission dataframe by copying column titles from sample submission file
sample_submission_dataframe = pd.read_csv(sample_submission_csv)
submission_dataframe = pd.DataFrame(columns=sample_submission_dataframe.columns)
submission_dataframe.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>toxic</th>
      <th>severe_toxic</th>
      <th>obscene</th>
      <th>threat</th>
      <th>insult</th>
      <th>identity_hate</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

logreg = LogisticRegression(C=6.0)
```


```python
for label in y_train:
    y = y_train[label]
    
    count = 0
    for i in y:
        if i == 1:
            count += 1
    print(count)
    
    logreg.fit(X_train_tokenized, y)
    
    y_pred = logreg.predict(X_train_tokenized)
    print("Training accuracy is {}".format(accuracy_score(y, y_pred))) 

    y_prob_test = logreg.predict_proba(X_test_tokenized)[:, 1]
    submission_dataframe[label] = y_prob_test 
```

    6762
    Training accuracy is 0.9882261703327693
    6303
    Training accuracy is 0.983909882810052
    12191
    Training accuracy is 0.9779250485680265
    1292
    Training accuracy is 0.9933963150968227
    1119
    Training accuracy is 0.994814188130601
    377
    Training accuracy is 0.9982687848593094



```python
for label in y_valid:
    y = y_valid[label]
    
    count = 0
    for i in y:
        if i == 1:
            count += 1
    print(count)
        
    y_pred = logreg.predict(X_valid_tokenized)
    print("Validation accuracy is {}".format(accuracy_score(y, y_pred)))  
```

    1687
    Validation accuracy is 0.9473288422371925
    1574
    Validation accuracy is 0.9510574964750117
    3103
    Validation accuracy is 0.9039009869967101
    303
    Validation accuracy is 0.9898167006109979
    286
    Validation accuracy is 0.990224032586558
    101
    Validation accuracy is 0.9972113426288579


# 7. View results


```python
# Prepare submission
submission_dataframe['id'] = test_dataframe['id'].tolist()
submission_dataframe.head(1)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>toxic</th>
      <th>severe_toxic</th>
      <th>obscene</th>
      <th>threat</th>
      <th>insult</th>
      <th>identity_hate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>00001cee341fdb12</td>
      <td>0.999915</td>
      <td>0.178272</td>
      <td>0.999215</td>
      <td>0.09397</td>
      <td>0.977419</td>
      <td>0.252907</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(test_dataframe.loc[submission_dataframe['toxic'] > 0.5].head(10))
```

                      id                                       comment_text
    0   00001cee341fdb12  Yo bitch Ja Rule is more succesful then you'll...
    7   000247e83dcc1211                   :Dear god this site is horrible.
    38  001068b809feee6b  " \n\n ==balance== \n This page has one senten...
    48  0013fed3aeae76b7  DJ Robinson is gay as hell! he sucks his dick ...
    50  001421530a1aa622  I have been perfectly civil in what quite clea...
    56  0016b94c8b20ffa6  I WILL BURN YOU TO HELL IF YOU REVOKE MY TALK ...
    59  0017d4d47894af05               :Fuck off, you anti-semitic cunt.  |
    63  00199e012d99a8b9  Her body is perfect. Face, boobs, hips, all of...
    70  001c86f5bceccb32  == Hello == \n\n Fuck off my Pagan you barebac...
    74  001d2f65ea6f4163  " August 2006 (UTC) \n\n :::::A simple ""you'r...



```python
print(test_dataframe.loc[submission_dataframe['id'] == '0016b94c8b20ffa6'].comment_text.values)
```

    ['I WILL BURN YOU TO HELL IF YOU REVOKE MY TALK PAGE ACCESS!!!!!!!!!!!!!']



```python
print(submission_dataframe.loc[submission_dataframe['id'] == '0016b94c8b20ffa6'])
```

                      id     toxic  severe_toxic   obscene    threat    insult  \
    56  0016b94c8b20ffa6  0.924701       0.07637  0.081055  0.357827  0.092477   
    
        identity_hate  
    56       0.008958  



```python
submission_dataframe.to_csv('submission.csv', index=False)
```
