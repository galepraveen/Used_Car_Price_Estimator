import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd
import warnings
import json
warnings.filterwarnings('ignore')

# the root path from where the pickle file will start and gets all the other pages
app = Flask(__name__)

# Loading the model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=["POST"])
def predict_api():
    data = request.json['data']
    print(data)

    input_values = list(data.values())
    print(input_values)

    test_data = np.array(input_values).reshape(1,-1)
    print(test_data)
    
    output = model.predict(test_data)
    output = float(output)
    print(output)

    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)