'''
Main API app
Enrique Marin
Jun 3rd 2023
'''
import sys
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
# import sklearn
# from sklearn.metrics import fbeta_score, precision_score, recall_score
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
import pandas as pd
import json
sys.path.append("./localremote/starter/starter/ml")
sys.path.append("./starter/ml")
from data import process_data  # noqa: E402
from model import inference  # noqa: E402


# Instantiate the app.
app = FastAPI()


# item to do inference from
class Item(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

    class Config:
        schema_extra = {"example": {
            "age": 21,
            "workclass": "State-gov",
            "fnlgt": 77516,
            "education": "Bachelors",
            "education_num": 13,
            "marital_status": "Never-married",
            "occupation": "Adm-clerical",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Male",
            "capital_gain": 217400000,
            "capital_loss": 0,
            "hours_per_week": 40,
            "native_country": "United-States"
            }
        }


# Define a GET on the specified endpoint.
@app.get("/")
async def say_hello():
    return {"greeting": """Welcome to the Deploying a Machine Learning Model
    with FastAPI project page"""}

@app.get("/api")
async def say_hello2():
    return {"greeting": """Welcome to the Deploying a Machine Learning Model
    with FastAPI project page"""}


@app.post("/run_model/")
async def run_inference_model(item: Item):
    age = item.age
    workclass = item.workclass
    fnlgt = item.fnlgt
    education = item.education
    education_num = item.education_num
    marital_status = item.marital_status
    occupation = item.occupation
    relationship = item.relationship
    race = item.race
    sex = item.sex
    capital_gain = item.capital_gain
    capital_loss = item.capital_loss
    hours_per_week = item.hours_per_week
    native_country = item.native_country
    # load model
    filename = './localremote/starter/model/model_rf_classifier_v1.pkl'
    loaded_model = joblib.load(open(filename, 'rb'))
    filename = './localremote/starter/model/model_rf_classifier_v1_encoder.pkl'
    loaded_encoder = joblib.load(open(filename, 'rb'))
    filename = './localremote/starter/model/model_rf_classifier_v1_lb.pkl'
    loaded_lb = joblib.load(open(filename, 'rb'))
    data_dict = {
        "age": age,
        "workclass": workclass,
        "fnlgt": fnlgt,
        "education": education,
        "education-num": education_num,
        "marital-status": marital_status,
        "occupation": occupation,
        "relationship": relationship,
        "race": race,
        "sex": sex,
        "capital-gain": capital_gain,
        "capital-loss": capital_loss,
        "hours-per-week": hours_per_week,
        "native-country": native_country,
        "salary": 'NA'
        }
    test = pd.DataFrame(columns=[
        'age', 'workclass', 'fnlgt',
        'education', 'education-num', 'marital-status', 'occupation',
        'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
        'hours-per-week', 'native-country', 'salary'])
    # test = test.append(data_dict, ignore_index = True)
    test = pd.concat([test, pd.DataFrame([data_dict])], ignore_index=True)
    print('test.head()')
    print(test.head())
    cat_features = [
        "workclass", "education", "marital-status",
        "occupation", "relationship", "race", "sex", "native-country"]
    X_test, y_test, encoder, lb = process_data(
        test, categorical_features=cat_features, label="salary",
        training=False, encoder=loaded_encoder, lb=loaded_lb)
    y_pred = inference(loaded_model, X_test)
    print('X_test')
    print(X_test)
    y_pred = inference(loaded_model, X_test)
    y_pred = y_pred[0]
    print('y_pred')
    print(y_pred)
    y_pred_label = lb.inverse_transform(y_pred)
    y_pred_label = y_pred_label[0]
    print('y_pred_label')
    print(y_pred_label)
    out = '{ "pred_number":"' + str(y_pred) + '","pred_label":"' + str(y_pred_label) + '"}'
    print(out)
    out = json.loads(out)
    return out
