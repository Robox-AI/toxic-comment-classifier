
# Getting started

**1. Download the project.**

```
git clone git@github.com:Robox-AI/toxic-comment-classifier.git
cd toxic-comment-classifier
```

**2. Create and activate a virtual environment.**

To keep dependencies local and easy to remove later, use virtualenv.

```
virtualenv .env
source .env/bin/activate
```

**3. Install project dependencies.**

This will install dependencies to your virtual environment's site-packages directory, e.g. `.env/lib/python3.6/site-packages/`.

```
pip install -r requirements.txt
```

**4. Download kaggle data.**

We'll do this using kaggle-cli, which was installed in the previous step. kaggle-cli makes it so we can download kaggle data without having to export kaggle site cookies and pass them to wget or cURL. It makes downloading kaggle data much easier.

```
mkdir download && cd download
kg download -u <kaggle_username> -p <kaggle_password> -c jigsaw-toxic-comment-classification-challenge
```

**5. Unarchive zipped kaggle data files.**

```
for f in *.zip; do tar zxvf "$f"; done
cd ..
```

**6. Open the notebook.**

```
jupyter notebook toxic-comments.ipynb
```

**7. Run the notebook inside Jupyter.**

**8. Deactivate virtual environment when finished.**

```
deactivate
```

# Jupyter notebook

[toxic-comments.ipynb](toxic-comments.ipynb)