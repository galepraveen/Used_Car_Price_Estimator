import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# the root path from where the pickle file will start and gets all the other pages
app = Flask(__name__, static_url_path='/assets', static_folder='assets')

# Loading the model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict/api', methods=["POST"])
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

@app.route('/predict', methods=["POST"])
def predict():
    data = list(request.form.values())
    data.pop(1)
    data = np.array(data).reshape(1,-1)
    print(data)
    output = model.predict(data)
    output = float(output)
    output = round(output, 5)

    return render_template('home.html', prediction_text = f'The car price predicted is around {output}')


if __name__ == "__main__":
    app.run(debug=True)