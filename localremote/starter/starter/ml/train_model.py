# Script to train machine learning model.

from sklearn.model_selection import train_test_split
import pandas as pd
import sys
import joblib

from ml.data import *
from ml.model import *  
# Add the necessary imports for the starter code.

# Add code to load in the data.
data = pd.read_csv('../data/census.csv')

data.columns = data.columns.str.strip()
# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=False, encoder = encoder, lb=lb
)

model = train_model(X_train, y_train)



y_pred = inference(model, X_test)

precision, recall, fbeta = compute_model_metrics(y_test, y_pred)

# save model
joblib.dump(model, './model/{}.pkl'.format(model_str_name))



