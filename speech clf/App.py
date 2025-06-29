# import necessary libraries
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
\
# create the flask app
app = Flask(__name__)
# load the model
model = pickle.load(open('model.pkl', 'rb'))
# load the vectorizer
vectorizer = pickle.load(open('cv.pkl', 'rb'))
# define the home route
@app.route('/')
def home():
    return render_template('index.html')
# define the predict route
@app.route('/predict', methods=['POST'])
def predict():
    # get the input text from the form
    input_text = request.form['text']
    # transform the input text using the vectorizer
    input_vector = vectorizer.transform([input_text])
    # make the prediction using the model
    prediction = model.predict(input_vector)
    # return the prediction as a JSON response
    return jsonify({'prediction': prediction[0]})
# run the flask app
if __name__ == '__main__':
    app.run(debug=True)