from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
dt_reg = joblib.load("models/dt_regressor.pkl")
dt_clf = joblib.load("models/dt_classifier.pkl")

@app.route("/")
def index():
    return render_template("index.html", title="Home")
    
@app.route("/predict", methods=["POST"])
def predict():
    gender = request.form["gender"]
    height = float(request.form["height"])  # allow decimals
    weight = float(request.form["weight"])  # allow decimals

    gender_num = 1 if gender.lower() == "male" else 0
    X_input = np.array([[gender_num, height, weight]])

    bmi_value = dt_reg.predict(X_input)[0]
    bmi_category = dt_clf.predict(X_input)[0]

    return render_template("result.html", bmi_value=round(bmi_value, 2), bmi_category=bmi_category)

@app.route("/insights")
def insights():
    return render_template("insights.html", title="Insights")

@app.route("/about")
def about():
    return render_template("about.html", title="About")

if __name__ == "__main__":
    app.run(debug=True)
