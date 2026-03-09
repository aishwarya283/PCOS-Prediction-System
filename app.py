from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("pcos_rf_model.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = [float(x) for x in request.form.values()]
        final = [np.array(data)]

        prediction = model.predict(final)
        probability = model.predict_proba(final)
        pcos_probability = probability[0][1] * 100

        if prediction[0] == 1:
            result = f"⚠ PCOS Detected\nRisk Probability: {pcos_probability:.2f}%"
        else:
            result = f"✅ No PCOS Detected\nRisk Probability: {pcos_probability:.2f}%"

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return f"Error: {e}"


import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
