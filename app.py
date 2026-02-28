from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load trained model and scaler
model = joblib.load("model.save")
scaler = joblib.load("transform.save")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():

    features = []

    for key in request.form:
        value = request.form[key]
        print(key, value)  # debug print
        features.append(float(value))

    print("Features received:", features)

    final_features = np.array([features])

    scaled_features = scaler.transform(final_features)

    prediction = model.predict(scaled_features)
    output = round(prediction[0], 2)

    return render_template(
        "index.html",
        prediction_text=f"Predicted Motor Temperature: {output} °C"
    )

if __name__ == "__main__":
    app.run(debug=True)