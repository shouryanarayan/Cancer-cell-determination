import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__) #Initialize the flask App
model = joblib.load('finalized_model2.sav')


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = np.array([int_features])
    prediction = model.predict(final_features)
    output = prediction[0]

    return render_template('index.html', prediction_text='Predicted class is {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)