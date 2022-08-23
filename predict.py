import pickle
from sklearn.metrics import accuracy_score, classification_report 

from flask import Flask
from flask import request
from flask import jsonify


model_file = 'models/model.bin'

with open(model_file, 'rb') as f_in:
    tfid, model = pickle.load(f_in)

app = Flask('NLP')

@app.route('/predict', methods=['POST'])
def predict():
    customer = request

    X = tfid.transform([customer])
    y_pred = model.predict(X)

    return y_pred