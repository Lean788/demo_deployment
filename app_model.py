from flask import Flask, request, jsonify
import os
import pickle
from sklearn.model_selection import cross_val_score
import pandas as pd
import sqlite3



os.chdir(os.path.dirname(__file__))

app = Flask(__name__)
app.config['DEBUG'] = True

@app.route("/", methods=['GET'])
def hello():
    return "Bienvenido a mi API del modelo advertising"

# 1. Wndpoint que devuelva la predicción de los nuevos datos enviados mediante argumentos en la llamada
@app.route('/v1/predict', methods=['GET'])
def predict():
    model = pickle.load(open('./data/advertising_model','rb'))

    tv = request.args.get('tv', None)
    radio = request.args.get('radio', None)
    newspaper = request.args.get('newspaper', None)

    if tv is None or radio is None or newspaper is None:
        return "Missing args, the input values are needed to predict"
    else:
        prediction = model.predict([[tv,radio,newspaper]])
        return "The prediction of sales investing that amount of money in TV, radio and newspaper is: " + str(round(prediction[0],2)) + 'k €'

@app.route("/v2/ingest_data", methods=['POST'])
def post_new_record():
    data = request.get_json()
    
    if not all(key in data for key in ['tv', 'radio', 'newspaper']):
        return 'Missing fields', 400
    
    tv = data['tv']
    radio = data['radio']
    newspaper = data['newspaper']
    
    try:
        with sqlite3.connect('./data/database.db') as connection:
            cursor = connection.cursor()
            cursor.execute('INSERT INTO database (tv, radio, newspaper) VALUES (?, ?, ?)', (tv, radio, newspaper))
            connection.commit()
            cursor.execute('SELECT COUNT(*) FROM database')
            count = cursor.fetchone()[0]
            connection.close()

    except Exception as e:
        return str(e), 500
    
    return f'Record added successfully. total records: {count}', 200


app.run()