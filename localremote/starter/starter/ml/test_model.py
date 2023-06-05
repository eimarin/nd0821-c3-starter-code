import pytest
import pandas as pd

from sklearn.model_selection import train_test_split
import sklearn
import numpy

import sys
sys.path.append("./localremote/starter/starter/ml")
sys.path.append("./starter/ml")
from data import process_data  # noqa: E402
from model import train_model, inference  # noqa: E402


@pytest.fixture(scope="session")
def data():
    df = pd.read_csv('localremote/starter/data/census.csv')
    return df


# # Optional: implement hyperparameter tuning.
def test_process_data():
    df = pd.read_csv('localremote/starter/data/census.csv')
    train, test = train_test_split(df, test_size=0.20)
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
    X_train, y_train, encoder, lb = process_data(train, categorical_features=cat_features, label="salary", training=True)
    assert isinstance(X_train, numpy.ndarray)
    assert isinstance(encoder, sklearn.preprocessing._encoders.OneHotEncoder)


def test_train_model():
    '''
    test the funciton train models
    '''
    print('test_train_model start')
    # df = data()
    # df = pd.read_csv('../../data/census.csv')
    df = pd.read_csv('localremote/starter/data/census.csv')
    train, test = train_test_split(df, test_size=0.20)
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
    print('--train_test_split section ok')
    X_train, y_train, encoder, lb = process_data(train, categorical_features=cat_features, label="salary", training=True)
    print('--process_data section ok')
    model = train_model(X_train, y_train)
    print('--train_model section ok')
    assert isinstance(model, sklearn.linear_model._logistic.LogisticRegression)


def test_inference():
    '''
    test the funciton inference models
    '''
    print('test_train_model start')
    # df = data()
    # df = pd.read_csv('../../data/census.csv')
    df = pd.read_csv('localremote/starter/data/census.csv')
    train, test = train_test_split(df, test_size=0.20)
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
    # clean
    print('--train_test_split section ok')
    res = process_data(train, categorical_features=cat_features,
                       label="salary", training=True)
    X_train, y_train, encoder, lb = res
    res = process_data(test, categorical_features=cat_features,
                       label="salary", training=False, encoder=encoder,
                       lb=lb)
    X_test, y_test, encoder, lb = res

    print('--process_data section ok')
    model = train_model(X_train, y_train)
    print('--train_model section ok')
    y_pred = inference(model, X_test)
    print('--train_model section ok')
    assert isinstance(y_pred, numpy.ndarray)
