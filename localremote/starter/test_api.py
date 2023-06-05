'''
latest-edit: 5 jun 23
'''
import requests
import json


def test_get():
    # response = requests.get('http://127.0.0.1:8000/')
    response = requests.get('https://test-udacity-app.onrender.com/')
    assert response.status_code == 200
    assert len(response.json()['greeting']) > 0


def test_post_1():
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
        "capital_gain": 2174,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
    }
    data = json.dumps(data)
    # response = requests.post('http://127.0.0.1:8000/run_model', data=data)
    response = requests.post('https://test-udacity-app.onrender.com/run_model', data=data)
    print(response)
    print(response.json())

    assert response.status_code == 200
    assert response.json()['pred_number'] == '0'


def test_post_2():
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
        "capital_gain": 217400000,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
    }
    data = json.dumps(data)

    # response = requests.post('http://127.0.0.1:8000/run_model', data=data)
    response = requests.post('https://test-udacity-app.onrender.com/run_model', data=data)
    print(response.json())

    assert response.status_code == 200
    assert response.json()['pred_number'] == '1'


if __name__=='__main__':
    test_post_1()
