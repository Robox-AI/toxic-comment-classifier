

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
X = train_dataframe['comment_text']
ys = train_dataframe[['obscene','insult','toxic','severe_toxic','identity_hate','threat']]

from sklearn.model_selection import train_test_split 
X_train, X_valid, y_train, y_valid = train_test_split(X, ys, test_size=0.2, random_state=1)
```

# 5 Tokenize words from comments


```python
from sklearn.feature_extraction.text import TfidfVectorizer

# create the transform
vectorizer = TfidfVectorizer()
# tokenize and build vocab
tfidf_matrix = vectorizer.fit_transform(X_train)
```


```python
# examine the vocabulary and document-term matrix together
pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names()).head()
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
if not open(f'vocab.pkl','wb'):
    pickle.dump(vectorizer, open(f'vocab.pkl','wb'))
```

# 6. Problem transformation

Train one binary classifier for each label. This is called binary relevance. 
