
# Getting started
**1. Download the project.**
```
git clone git@github.com:Robox-AI/toxic-comment-classifier.git
cd toxic-comment-classifier
```
**2. To keep dependencies local and easy to remove later, use virtualenv.**
```
virtualenv .env
source .env/bin/activate
```
**3. Install this notebook's dependencies.**
```
pip install -r requirements.txt
```
**4. Open the notebook.**
```
jupyter notebook toxic-comments.ipynb
```
**5. Run the notebook inside Jupyter.**

# Notebook contents


```python
# Reload modules before executing user code
%reload_ext autoreload
# Reload all modules (except those excluded by %aimport)
%autoreload 2
# Show plots within this notebook
%matplotlib inline
```

## Load training and test data into pandas dataframes


```python
PATH='download/'
test_csv = f'{PATH}test.csv'
train_csv = f'{PATH}train.csv'
sample_submission_csv = f'{PATH}sample_submission.csv'
```


```python
import pandas as pd

train_df = pd.read_csv(train_csv, na_filter=False)
test_df = pd.read_csv(test_csv, na_filter=False)
submission_df = pd.read_csv(sample_submission_csv, nrows=0) # copy column headers
```

## Explore the data

The labels are all in the same scale and won't need to be standardized. Notice how a comment can have multiple labels, e.g. the comment below is both toxic and a threat. This looks like a multilabel text classification problem, which can be solved in a variety of ways.

**(1) Problem transformation methods**

Problem transformation transforms the multilabel input into a representation suitable for single-label classification methods.

* **Binary Relevance** - Independently train one binary classifier for each label. The drawback of this method is that it does not take into account label correlation.

* **Label Powerset** - Generate a new class for every combination of labels and then use multiclass classification. Unlike binary relevance, this method takes into account label correlation, but it leads to a large number of classes and fewer examples per class.

* **Classifier Chains** - Based on Binary Relevance but predictions of binary classifiers are cascaded along a chain as additional features. This method takes into account label correlation but the order of classifiers in the chain changes results.

**(2) Algorithm adaptation methods**

Algorithm adaption extends existing single-label classifier algorithms to handle multilabel data directly.


```python
train_df.loc[train_df['threat'] == 1].head(1)
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
train_df.shape
```




    (159571, 8)




```python
train_df.describe()
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
train_df.info() # verify that are no missing values in our dataset
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


## Separate target features (y) from input features (X)


Use sklearn.model_selection.train_test_split to split training data into validation and train.


```python
from sklearn.model_selection import train_test_split

X = train_df['comment_text']
y = train_df[['obscene','insult','toxic','severe_toxic','identity_hate','threat']]

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=1)
X_test = test_df['comment_text']
```

## Create a TF-IDF matrix

Count how many times each word appears in the comments (term frequency) and multiply it by the context-adjusted weight of each word (inverse document frequency). Better explained here: https://www.quora.com/How-does-TfidfVectorizer-work-in-laymans-terms


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
dt_matrix = pd.DataFrame(X_train_tokenized.toarray(), columns=vectorizer.get_feature_names())
```


```python
dt_matrix.head(1)
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
  </tbody>
</table>
<p>1 rows × 165609 columns</p>
</div>




```python
dt_matrix.head(1).loc[:, (dt_matrix.head(1) != 0).any(axis=0)]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>allowed</th>
      <th>attack</th>
      <th>be</th>
      <th>blocked</th>
      <th>but</th>
      <th>comments</th>
      <th>definitely</th>
      <th>editor</th>
      <th>editors</th>
      <th>if</th>
      <th>...</th>
      <th>that</th>
      <th>the</th>
      <th>their</th>
      <th>they</th>
      <th>this</th>
      <th>to</th>
      <th>ve</th>
      <th>while</th>
      <th>will</th>
      <th>won</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.200271</td>
      <td>0.184008</td>
      <td>0.154242</td>
      <td>0.299486</td>
      <td>0.088722</td>
      <td>0.159946</td>
      <td>0.21692</td>
      <td>0.165467</td>
      <td>0.156545</td>
      <td>0.171023</td>
      <td>...</td>
      <td>0.066082</td>
      <td>0.049339</td>
      <td>0.266064</td>
      <td>0.223291</td>
      <td>0.07312</td>
      <td>0.161087</td>
      <td>0.133756</td>
      <td>0.15515</td>
      <td>0.105123</td>
      <td>0.18268</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 30 columns</p>
</div>



## Problem transformation

Train one binary classifier for each label. This is called binary relevance.


```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

logreg = LogisticRegression(C=6.0)
```


```python
for label in y_train:
    y = y_train[label]
    logreg.fit(X_train_tokenized, y)
    y_pred = logreg.predict(X_train_tokenized)
    print("Training accuracy for {} comments is {}".format(label, accuracy_score(y, y_pred)))
    y_prob_test = logreg.predict_proba(X_test_tokenized)[:, 1]
    submission_df[label] = y_prob_test
```

    Training accuracy for obscene comments is 0.9882261703327693
    Training accuracy for insult comments is 0.983909882810052
    Training accuracy for toxic comments is 0.9779250485680265
    Training accuracy for severe_toxic comments is 0.9933963150968227
    Training accuracy for identity_hate comments is 0.994814188130601
    Training accuracy for threat comments is 0.9982687848593094



```python
for label in y_valid:
    y = y_valid[label]
    y_pred = logreg.predict(X_valid_tokenized)
    print("Validation accuracy for {} comments is {}".format(label, accuracy_score(y, y_pred)))
```

    Validation accuracy for obscene comments is 0.9473288422371925
    Validation accuracy for insult comments is 0.9510574964750117
    Validation accuracy for toxic comments is 0.9039009869967101
    Validation accuracy for severe_toxic comments is 0.9898167006109979
    Validation accuracy for identity_hate comments is 0.990224032586558
    Validation accuracy for threat comments is 0.9972113426288579


## View results


```python
# Prepare submission
submission_df['id'] = test_df['id'].tolist()
submission_df.head(1)
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
print(test_df.loc[submission_df['toxic'] > 0.5].head(10))
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
print(test_df.loc[submission_df['id'] == '0016b94c8b20ffa6'].comment_text.values)
```

    ['I WILL BURN YOU TO HELL IF YOU REVOKE MY TALK PAGE ACCESS!!!!!!!!!!!!!']



```python
print(submission_df.loc[submission_df['id'] == '0016b94c8b20ffa6'])
```

                      id     toxic  severe_toxic   obscene    threat    insult  \
    56  0016b94c8b20ffa6  0.924701       0.07637  0.081055  0.357827  0.092477

        identity_hate
    56       0.008958


^ That looks about right

## Save results to CSV for submission


```python
submission_df.to_csv('submission.csv', index=False)
```
