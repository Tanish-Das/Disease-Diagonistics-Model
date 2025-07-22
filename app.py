from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)
model, mlb = pickle.load(open("model/disease_model.pkl", "rb"))

precaution_df = pd.read_csv("Disease precaution.csv")
precaution_dict = {
    row["Disease"]: [row["Precaution_1"], row["Precaution_2"], row["Precaution_3"], row["Precaution_4"]]
    for _, row in precaution_df.iterrows()
}

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    precautions = []
    symptoms = mlb.classes_

    if request.method == "POST":
        user_symptoms = request.form.getlist("symptoms")
        user_input = mlb.transform([user_symptoms])
        prediction = model.predict(user_input)[0]
        precautions = precaution_dict.get(prediction, ["No precaution data available."])

    return render_template("index.html", symptoms=symptoms, prediction=prediction, precautions=precautions)

if __name__ == "__main__":
    app.run(debug=True)
