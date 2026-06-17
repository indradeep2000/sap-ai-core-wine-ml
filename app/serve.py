from flask import Flask, request, jsonify
import pickle
import os
import numpy as np


app = Flask(__name__)

# Load trained model
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")

with open(model_path, "rb") as file:
    model = pickle.load(file)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Wine ML inference API is running Indra",
        "status": "OK"
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        features = data.get("features")

        if features is None:
            return jsonify({
                "error": "Missing 'features' in request body"
            }), 400

        # Convert input into numpy array
        input_data = np.array(features).reshape(1, -1)

        prediction = model.predict(input_data)

        return jsonify({
            "prediction": int(prediction[0])
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)