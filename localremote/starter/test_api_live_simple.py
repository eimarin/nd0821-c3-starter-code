import requests
import pytest
import json

data = {
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
	"capital_gain":2174,
	"capital_loss": 0,
	"hours_per_week": 40,
	"native_country": "United-States"
	}
data = json.dumps(data)
# response = requests.post('https://test-udacity-app.onrender.com:10000/run_model', data=data)

response = requests.get('https://test-udacity-app.onrender.com:22449/')


# response = requests.post('http://127.0.0.1:8000/run_model', data=data)
print(response.json())
print(response)

assert response.status_code == 200
# assert response.json()['pred_number'] == '0'