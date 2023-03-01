from flask import Flask, request, jsonify
import os
import pickle
from sklearn.model_selection import cross_val_score
import pandas as pd
import sqlite3
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV



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
    
    if not all(key in data for key in ['tv', 'radio', 'newspaper', 'sales']):
        return 'Missing fields', 400
    
    tv = data['tv']
    radio = data['radio']
    newspaper = data['newspaper']
    sales = data['sales']

    connection = sqlite3.connect('./data/database.db')
    cursor = connection.cursor()
    cursor.execute('INSERT INTO database (tv, radio, newspaper, sales) VALUES (?, ?, ?, ?)', (tv, radio, newspaper, sales)).fetchall()
    cursor.execute('SELECT COUNT(*) FROM database')
    count = cursor.fetchone()[0]
    connection.commit()
    connection.close()
    
    return f'Record added successfully. The records in Database is {count}', 200

@app.route("/get_all/", methods=['GET'])
def get_all():
    connection = sqlite3.connect('./data/database.db')
    cursor = connection.cursor()
    select_all = "SELECT * FROM database"
    result = cursor.execute(select_all).fetchall()
    connection.close()
    return result

@app.route('/v2/retrain', methods=['GET'])
def retrain():
    from sklearn.model_selection import GridSearchCV
    model = pickle.load(open('./data/advertising_model','rb'))
    connection = sqlite3.connect('./data/database.db')
    cursor = connection.cursor()
    cursor.execute('SELECT COUNT(*) FROM database')
    count = cursor.fetchone()[0]
    cursor.execute("SELECT * FROM database")
    data = cursor.fetchall()

    df = pd.DataFrame(data, columns=['id', 'tv', 'radio', 'newspaper', 'sales'])

    X = df.iloc[:, 1:-1]
    y = df.iloc[:, -1]

    n_train = round(((len(df))*0.8), 0)

    # train : las primeras filas 
    # test : las últimas filas
    n_split = int(round(((len(df))*0.8), 0))

    X_train = df.iloc[:n_split,1:-1] # sin sales ni la columna id
    X_test = df.iloc[n_split:,1:-1]
    y_train = df.iloc[:n_split,-1] # columna sales
    y_test = df.iloc[n_split:,-1]  # columna sales

    def retrain_model(model, X, y, alpha=5500):
        """
        Esta función reentrena un modelo existente con datos existentes y nuevos, entendiendose que los nuevos
        estarán al final de la tabla.

        Args:
        - model: el modelo a reentrenar.
        - X: matriz con los datos de características.
        - y: vector con las etiquetas de los datos.
        - alpha: el parámetro de regularización para el modelo Ridge.

        Returns:
        - El modelo reentrenado.
        """
        from sklearn.linear_model import Ridge
        
        # Reentrenar el modelo
        model.set_params(alpha=alpha)
        model.fit(X, y)

        return model
    
    new_model = retrain_model(model, X, y)

    # Comparar el accuracy del nuevo modelo con el modelo existente
    existing_accuracy = model.score(X, y)
    new_accuracy = new_model.score(X, y)

    # Buscar los mejores hiperparámetros usando Grid Search
    params = {'alpha': [5500, 6000, 7000]}
    grid_search = GridSearchCV(Ridge(), params, scoring='neg_mean_squared_error', cv=5)
    grid_search.fit(X_train, y_train)
    new_model = grid_search.best_estimator_
    new_accuracy = new_model.score(X_test, y_test)

    # Actualizar el mensaje a retornar dependiendo de si el nuevo modelo es mejor o no
    if new_accuracy > existing_accuracy:
        pickle.dump(new_model, open('./data/advertising_model', 'wb'))
        model = new_model
        result = {"status": "success","database_size":f"{count}", "message": "Se ha reentrenado el modelo con éxito y se ha guardado el nuevo modelo porque tiene un mejor rendimiento."}
    else:
        result = {"status": "success","database_size":f"{count}", "message": "Se ha reentrenado el modelo con éxito, pero se sigue utilizando el modelo anterior porque tras el reentreno ha obtenido iguales o peores resultados."}

    connection.close()

    # Retornar el mensaje
    return result


app.run()