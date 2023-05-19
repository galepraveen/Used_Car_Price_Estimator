import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# the root path from where the pickle file will start and gets all the other pages
app = Flask(__name__, static_url_path='/static', static_folder='static')

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
    output = round(output, 2)

    print(output)

    return jsonify(output)

@app.route('/predict', methods=["POST"])
def predict():
    print('\n\n')
    data = list(request.form.values())
    print(data)
    data.pop(1)
    print(data)
    data = np.array(data).reshape(1,-1)
    print(data)
    output = model.predict(data)
    output = float(output)
    output = round(output, 5)

    return render_template('home.html', prediction_text = f'The Car Price Predicted Is { output } Lakhs')


if __name__ == "__main__":
    app.run(debug=True)