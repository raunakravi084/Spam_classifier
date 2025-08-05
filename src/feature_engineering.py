import numpy as np
import pandas as pd

import os

from sklearn.feature_extraction.text import CountVectorizer

# fetch the data from data/processed
train_data = pd.read_csv('./data/processed/train_processed.csv')
test_data = pd.read_csv('./data/processed/test_processed.csv')

train_data.fillna('',inplace=True)
test_data.fillna('',inplace=True)

# Splitting the data into features and target variable
X_train = train_data['Text'].values
y_train = train_data['Target'].values

X_test = test_data['Text'].values
y_test = test_data['Target'].values



