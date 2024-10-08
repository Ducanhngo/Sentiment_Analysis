# -*- coding: utf-8 -*-
"""Sentiment_Analysis.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1GYU8ABS8bxCWn-LfHR7ZATXiYeiQIZxn
"""

import pandas as pd
df = pd.read_csv("/content/IMDB-Dataset.csv")
df = df.drop_duplicates()

!pip install -q contractions

import re
import string
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
import contractions

stop = set(stopwords.words('english'))

def expand_contractions(text):
    return contractions.fix(text)

def prepocessing_text(text):
  wl = WordNetLemmatizer()
  soup = BeautifulSoup(text, "html.parser")
  text = soup.get_text()
  text = expand_contractions(text)
  emoji_clean = re.compile("["
                          u"\U0001F600-\U0001F64F"
                          u"\U0001F300-\U0001F5FF"
                          u"\U0001F680-\U0001F6FF"
                          u"\U0001F1E0-\U0001F1FF"
                          u"\U00002702-\U000027B0"
                          u"\U000024C2-\U0001F251"
                          "]+", flags=re.UNICODE)
  text = emoji_clean.sub(r'', text)
  text = re.sub(r"\.(?=S)", ". ", text)
  text = re.sub(r"http\S+", "", text)
  text = "".join([wl.lemmatize(word) for word in text if word not in string.punctuation])
  text = " ".join([word for word in text.split() if word not in stop and word.isalpha()])
  return text

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def funf(pct, allvalues):
  absolute = int(np.round(pct/100.*np.sum(allvalues)))
  return "{:.1f}%\n({:d} g)".format(pct, absolute)

freg_pos = len(df[df["sentiment"] == "positive"])
freg_neg = len(df[df["sentiment"] == "negative"])

data = [freg_pos, freg_neg]

labels = ['positive', 'negative']

pie, ax = plt.subplots(figsize=[11, 7])
plt.pie(x=data, autopct = lambda pct: funf(pct, data), explode = [0.0025]*2, pctdistance = 0.5, colors = [sns.color_palette()[0], 'tab:red'], textprops={'fontsize':16})
labels = [r'Positive', r'Negative']
plt.legend(labels, loc = 'best', prop={'size':14})
plt.show()

words_len = df['review'].str.split().map(lambda x: len(x))
df_temp = df.copy()
df_temp['words_len'] = words_len

hist_positive = sns.displot(
    data=df_temp[df_temp['sentiment'] == 'positive'],
    x='words_len', hue = 'sentiment', kde=True, height = 7, aspect = 1.1, legend = False
).set(title = "Words in postive reviews")
plt.show(hist_positive)

his_negative = sns.displot(
    data=df_temp[df_temp['sentiment'] == 'negative'],
    x='words_len', hue = 'sentiment', kde=True, height = 7, aspect = 1.1, legend = False, palette = ['red']
).set(title = "Words in negative reviews")
plt.show(his_negative)

plt.figure(figsize=(7, 7.1))
kernel_distribution_number_words_plot = sns.kdeplot(
    data=df_temp, x = 'words_len', hue = 'sentiment', fill = True, palette = [sns.color_palette()[0],'red']
).set(title='Words in reviews')
plt.show(kernel_distribution_number_words_plot)

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
y_data = label_encoder.fit_transform(df['sentiment'])
x_data = df['review'].apply(prepocessing_text)
x_train, x_test, y_train, y_test = train_test_split(
    x_data,  y_data, test_size=0.2, random_state=42
)

tfidf_vectorizer = TfidfVectorizer(max_features = 10000)
tfidf_vectorizer.fit(x_train, y_train)

x_train_encoded = tfidf_vectorizer.transform(x_train)
x_test_encoded = tfidf_vectorizer.transform(x_test)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

dt_classifier = DecisionTreeClassifier(
    criterion = 'entropy',
    random_state = 42
)

dt_classifier.fit(x_train_encoded, y_train)
y_pred = dt_classifier.predict(x_test_encoded)
accuracy = accuracy_score(y_test, y_pred)
rf_classifier = RandomForestClassifier(
    random_state = 42
)

rf_classifier.fit(x_train_encoded, y_train)
y_pred = rf_classifier.predict(x_test_encoded)
accuracy = accuracy_score(y_test, y_pred)