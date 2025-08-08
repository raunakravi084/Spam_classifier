import numpy as np
import pandas as pd

import pickle
import json
import os
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score
import warnings
warnings.filterwarnings("ignore")

clf = pickle.load(open('data/model/model.pkl','rb'))
test_data = pd.read_csv('./data/features/test_bow.csv')

# Drop the label column for features
X_test = test_data.drop(columns= 'label', axis=1).values
y_test = test_data['label'].values

y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)[:, 1]

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

metrics_dict={
    'accuracy':accuracy,
    'precision':precision,
    'recall':recall,
    'auc':auc
}

# store the data inside data/features
data_path = os.path.join("data","evaluation")

os.makedirs(data_path,exist_ok=True)

print(f"{'<'*20} Inside model_evaluation, Saving the files {'>'*20}")

with open(os.path.join(data_path,"metrics.json"), 'w') as file:
    json.dump(metrics_dict, file, indent=4)