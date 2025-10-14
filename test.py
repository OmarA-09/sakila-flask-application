import tensorflow as tf
import numpy as np
from flask import request, jsonify
from api.models import Film

@films_router.post('/neuralnet/train')
def train_film_neural_network():
    films = Film.query.all()
    x_train = np.array([
        [f.release_year, f.rental_duration, f.rental_rate, f.length, f.replacement_cost]
        for f in films
    ])
    y_train = np.array([f.length for f in films])
    x_train = x_train / np.max(x_train, axis=0)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(x_train.shape[1],)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.fit(x_train, y_train, epochs=20, verbose=2)
    model.save("api/neural_net_models/sakila_film_nn_model.h5")
    return jsonify({"message": "Neural network model trained and saved."}), 201

@films_router.post('/neuralnet/predict')
def predict_film_neural_network():
    model = tf.keras.models.load_model("api/neural_net_models/sakila_film_nn_model.h5")
    data = request.get_json()
    features = np.array(data['features'])
    features = features / np.max(features, axis=0)
    predictions = model.predict(features)
    return jsonify({"predictions": predictions.tolist()}), 200
