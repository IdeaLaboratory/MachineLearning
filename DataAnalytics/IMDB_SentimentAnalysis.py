import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import style as stl
from statsmodels.tsa.arima_model import ARIMA
import seaborn as sns
import sklearn
import os
import string
import sklearn.datasets
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import adjusted_mutual_info_score
from collections import Counter
import re
import plotly.plotly as pyp
import operator
import time
from sklearn.feature_extraction.text import CountVectorizer
#from wordcloud import WordCloud
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
print('Import successfull')

# data cleaning


def remove_stop_words(df):
    stop_words = set(stopwords.words('english'))
    count = 0
    for sentance in df:
        sentance = [word for word in sentance.lower().split()
                    if word not in stop_words]
        sentance = ' '.join(sentance)
        df[count] = sentance
        count += 1
        return df


def remove_puntuation(df):
    count = 0
    for s in df:
        clean = re.compile('<.*?>')
        s = re.sub(r'\d+', '', s)
        s = re.sub(clean, '', s)
        s = re.sub("'", '', s)
        s = re.sub(r'\W+''', s)
        s = re.sub('_', '', s)
        df[count] = s
        count += 1
        return df

# lematization => not sentax but simantax


def lemma(df):
    lmtzr = WordNetLemmatizer()
    count = 0
    stemmed = []
    for s in df:
        word_token = word_tokenize(s)
        for word in s:
            stemmed.append(lmtzr.lemmatize(word))
            s = ' '.join(stemmed)
            df.iloc[count] = s
            count += 1
            stemmed = []
        return df


def stemmer(df):

    stm = SnowballStemmer("english", ignore_stopwords=True)
    count = 0
    stemmed = []
    for sentence in df:
        word_tokens = word_tokenize(sentence)
        for word in word_tokens:
            stemmed.append(stem.stem(word))
        sentence = ' '.join(stemmed)
        df.iloc[count] = sentence
        count = count + 1
        stemmed = []
    return df


# data processing
df = pd.read_csv('imdb_master.csv', encoding='latin-1', index_col=0)
df.head(20)
df.tail(20)
df = df[df.label != 'unsup']
df_train = df[df.type == 'train']
df_train = df_train[:15000]
df_test = df[df.type == 'test']
df_test = df_test[:5000]

y_train = df_train.label
y_test = df_test.label
df_train.review = remove_stop_words(df_train.review)
df_test.review = remove_stop_words(df_test.review)
print(df_test)