from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('hackathon.html')

@app.route('/predict', methods=['POST', 'GET'])
def result():
    Education = float(request.form['Education'])
    no_of_trainings = float(request.form['no_of_trainings'])
    age = float(request.form['age'])
    previous_year_rating = float(request.form['previous_year_rating'])
    length_of_service = float(request.form['length_of_service'])
    KPIs_met = float(request.form['KPIs_met'])
    Awards = float(request.form['Awards'])
    avg_training_score = float(request.form['avg_training_score'])

    X = np.array([[Education, no_of_trainings, age, previous_year_rating, length_of_service,
                   KPIs_met, Awards, avg_training_score]])
    model = pickle.load(open('hackathon.pkl', 'rb'))
    y_predict = model.predict(X)
    return jsonify({'Prediction': float(y_predict)})


if __name__ == "__main__":
    app.run(debug=True, port=1234)
