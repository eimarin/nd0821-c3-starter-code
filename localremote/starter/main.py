# Put the code for your API here.
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
import pandas as pd
from starter.ml.data import *
from starter.ml.model import *
import json


# Instantiate the app.
app = FastAPI()

# item to do inference from
class Item(BaseModel):
    age: int
    workclass: str
    fnlgt:int
    education:str
    education_num:int
    marital_status:str
    occupation:str
    relationship:str
    race:str
    sex:str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country:str

# Define a GET on the specified endpoint.
@app.get("/")
async def say_hello():
    return {"greeting": "Welcome to the Deploying a Machine Learning Model with FastAPI project page "+str(pd.__version__) }

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
	filename = './model/model_rf_classifier_v1.pkl'
	loaded_model = joblib.load(open(filename, 'rb'))
	filename = './model/model_rf_classifier_v1_encoder.pkl'
	loaded_encoder = joblib.load(open(filename, 'rb'))
	filename = './model/model_rf_classifier_v1_lb.pkl'
	loaded_lb = joblib.load(open(filename, 'rb'))

	print(type(loaded_model))
	print(type(loaded_encoder))
	print(type(loaded_lb))

	data_dict  = {
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
		"capital-gain":capital_gain,
		"capital-loss": capital_loss,
		"hours-per-week": hours_per_week,
		"native-country": native_country,
		"salary":'NA'
		}
			
	# age,workclass,fnlgt,education,education-num,marital-status,occupation,relationship,race,sex,capital-gain,capital-loss,hours-per-week,native-country,salary
	# columns=['age','workclass','fnlgt','education','education_num','marital_status','occupation','relationship','race','sex','capital_gain','capital_loss','hours_per_week','native_country']
	test = pd.DataFrame(columns=['age','workclass','fnlgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','salary'])
	# test = test.append(data_dict, ignore_index = True)
	test = pd.concat([test, pd.DataFrame([data_dict])], ignore_index=True)
	print('test.head()')
	print(test.head())
	cat_features = [ "workclass", "education", "marital-status", "occupation", "relationship", "race", "sex","native-country"]
	X_test, y_test, encoder, lb = process_data(test, categorical_features=cat_features, label="salary", training=False, encoder = loaded_encoder, lb=loaded_lb)
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
	out =  '{ "pred_number":"'+ str(y_pred) + '","pred_label":"' + str(y_pred_label) + '"}'
	print(out)
	out = json.loads(out)
	return out
	# return str(y_pred) + "- "+ str(y_pred_label)