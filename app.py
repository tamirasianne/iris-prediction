from flask import Flask, render_template, request
import numpy as np
import pickle
import pandas as pd

app = Flask(__name__)

model = pickle.load(open("rf_model.pkl", "rb"))

species = ["Iris setosa", "Iris versicolor", "Iris virginica"]

LIMITS = {
    "sepal_length": (4.0, 8.0),
    "sepal_width": (2.0, 4.5),
    "petal_length": (1.0, 7.0),
    "petal_width": (0.1, 2.6)
}

def validate_input(name, value):
    min_val, max_val = LIMITS[name]
    if not (min_val <= value <= max_val):
        raise ValueError(
            f"{name.replace('_',' ').title()} must be between {min_val} and {max_val}"
        )

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        sepal_length = float(request.form["sepal_length"])
        sepal_width = float(request.form["sepal_width"])
        petal_length = float(request.form["petal_length"])
        petal_width = float(request.form["petal_width"])

        validate_input("sepal_length", sepal_length)
        validate_input("sepal_width", sepal_width)
        validate_input("petal_length", petal_length)
        validate_input("petal_width", petal_width)


        features = pd.DataFrame([{
            "sepal length (cm)": sepal_length,
            "sepal width (cm)": sepal_width,
            "petal length (cm)": petal_length,
            "petal width (cm)": petal_width
        }])

        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]

        result = species[prediction]
        confidence = round(float(np.max(probabilities)) * 100, 2)

        return render_template(
            "index.html",
            prediction_text=f"Predicted Species: {result}",
            confidence_text=f"Confidence: {confidence}%"
        )

    except ValueError as err:
        return render_template("index.html", prediction_text=str(err))

    except KeyError as err:
        return render_template("index.html", prediction_text=f"Missing field: {err}")

    except TypeError as err:
        return render_template("index.html", prediction_text=f"Invalid input type: {err}")

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)