import pytest
import pandas as pd


from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from data import *
from model import *


@pytest.fixture(scope="session")
def data():
    df = pd.read_csv('../../data/census.csv')

    return df

# Optional: implement hyperparameter tuning.
def test_process_data(process_data):
    df = data()
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
    X_train, y_train, encoder, lb = process_data( train, categorical_features=cat_features, label="salary", training=True)
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(encoder, pd.DataFrame)

def test_train_model(train_model):
    '''
    test the funciton train models
    '''

    df = data()
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
    X_train, y_train, encoder, lb = process_data( train, categorical_features=cat_features, label="salary", training=True)
    model = train_model(X_train, y_train)
    assert isinstance(model, pd.DataFrame)

def test_train_model(train_model):
    '''
    test the funciton train models
    '''

    df = data()
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
    X_train, y_train, encoder, lb = process_data( train, categorical_features=cat_features, label="salary", training=True)
    model = train_model(X_train, y_train)
    assert isinstance(model, pd.DataFrame)

  