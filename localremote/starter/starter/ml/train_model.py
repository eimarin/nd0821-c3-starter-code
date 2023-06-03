# Script to train machine learning model.

from sklearn.model_selection import train_test_split
import pandas as pd
import sys
import joblib

from data import *
from model import *  
# Add the necessary imports for the starter code.

# Add code to load in the data.


def metrics_based_slice(model, test, slice_column, encoder, lb):
    value_list = test[slice_column].unique().tolist()
    out_str = ''
    for value in value_list:
        mask = test[slice_column] == value
        test_iteration = test[mask]
        X_test, y_test, encoder, lb = process_data(
            test_iteration, categorical_features=cat_features, label="salary", training=False, encoder = encoder, lb=lb
            )
        y_pred = inference(model, X_test)
        precision, recall, fbeta = compute_model_metrics(y_test, y_pred)
        out_str += slice_column + ' ' + value + '\n'
        out_str += 'precision,' + str(precision) + '\n'
        out_str += 'recall,' + str(recall) + '\n'
        out_str += 'fbeta,' + str(fbeta) + '\n'
        out_str += 38*'-' + '\n'
    return out_str



data = pd.read_csv('./data/census.csv')

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
    test, categorical_features=cat_features, label="salary", training=False, encoder = encoder, lb=lb
)

model = train_model(X_train, y_train)
y_pred = inference(model, X_test)

precision, recall, fbeta = compute_model_metrics(y_test, y_pred)
print('precision,',str(precision))
print('recall,',str(recall))
print('fbeta,',str(fbeta))

model_str_name = 'model_rf_classifier_v1'
# save model
joblib.dump(model, './model/{}.pkl'.format(model_str_name))

# save enconder
joblib.dump(encoder, './model/{}_encoder.pkl'.format(model_str_name))

# save lb
joblib.dump(lb, './model/{}_lb.pkl'.format(model_str_name))


# https://knowledge.udacity.com/questions/909273
with open('slice_output.txt', 'w') as out:
    for slice_column in cat_features:
        out_str = 38*'#'+ '\n'
        out_str += 'OVERALL'+ '\n'
        out_str += 38*'#'+ '\n'
        out_str += 'precision,' + str(precision) + '\n'
        out_str += 'recall,' + str(recall) + '\n'
        out_str += 'fbeta,' + str(fbeta) + '\n'
        out_str += 38*'-' + '\n'
        out.write(out_str)

        out.write(38*'#'+ '\n')
        out.write(slice_column+ '\n')
        out.write(38*'#'+ '\n')
        out_str =  metrics_based_slice(model, test, slice_column, encoder, lb)
        out.write(out_str)