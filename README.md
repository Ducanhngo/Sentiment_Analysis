# Sentiment Analysis Project

This project focuses on sentiment analysis using the IMDB movie review dataset. It explores the use of Decision Tree and Random Forest classifiers to classify reviews as positive or negative.


## Acknowledgments
I want to express my heartfelt thanks to [AI VIET NAM](https://aivietnam.edu.vn/) for their incredible support and guidance throughout this project. Their assistance has been invaluable in making this project a success.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Text Preprocessing](#text-preprocessing)
- [Model Training](#model-training)
- [Evaluation](#evaluation)


## Introduction
In this project, we analyze customer sentiment through movie reviews. The goal is to classify the reviews into positive or negative sentiments using various machine learning models.

## Dataset
The dataset used in this project is the IMDB Movie Review Dataset. It consists of labeled reviews which are classified into two categories:
- **Positive**
- **Negative**

Download the dataset from [this link](https://drive.google.com/uc?id=1v36q7Efz0mprjAv4g6TkQM2YlDKdqOuy).

## Text Preprocessing
Before feeding the data into the models, we clean and preprocess the text using the following steps:
1. Removing HTML tags
2. Expanding contractions
3. Removing emoticons, symbols, and punctuations
4. Lemmatizing words
5. Removing stopwords and non-alphabetic tokens

```python
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import contractions

# Load and clean data
df = pd.read_csv('IMDB-Dataset.csv')
df['review'] = df['review'].apply(preprocess_text)
```

## Model Training
We use two machine learning models to classify the reviews:
- **Decision Tree**
- **Random Forest**

Both models are trained on the processed text data, and the `TfidfVectorizer` is used for converting the text into numerical features.

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Splitting the data
x_train, x_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)

# Text vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=10000)
x_train_encoded = tfidf_vectorizer.fit_transform(x_train)
x_test_encoded = tfidf_vectorizer.transform(x_test)

# Decision Tree
dt_classifier = DecisionTreeClassifier(criterion='entropy', random_state=42)
dt_classifier.fit(x_train_encoded, y_train)

# Random Forest
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(x_train_encoded, y_train)
```

## Evaluation
After training the models, we evaluate their performance using accuracy scores.

```python
from sklearn.metrics import accuracy_score

# Evaluate Decision Tree
y_pred_dt = dt_classifier.predict(x_test_encoded)
accuracy_dt = accuracy_score(y_pred_dt, y_test)

# Evaluate Random Forest
y_pred_rf = rf_classifier.predict(x_test_encoded)
accuracy_rf = accuracy_score(y_pred_rf, y_test)
```

Here are the key packages:
- `pandas`
- `scikit-learn`
- `nltk`
- `beautifulsoup4`
- `contractions`
- `seaborn`

## Usage
1. Download the IMDB dataset.
2. Preprocess the dataset.
3. Train the models.
4. Evaluate the performance.
