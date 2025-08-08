import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
import os
import warnings
warnings.filterwarnings("ignore")

# fetch the data from data/processed
train_data = pd.read_csv('./data/features/train_bow.csv')

# Drop the label column for features
X_train = train_data.drop(columns= 'label', axis=1).values
y_train = train_data['label'].values

# Define and train the LogisticRegression model

clf = LogisticRegression()
clf.fit(X_train, y_train)

# store the data inside data/features
data_path = os.path.join("data","model")
os.makedirs(data_path,exist_ok=True)

print(f"{'<'*20} Inside model_building, Saving the files {'>'*20}")

# save
pickle.dump(clf, open(os.path.join(data_path,"model.pkl"),'wb'))
