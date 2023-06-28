
from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__, template_folder='C:\\Users\\Ziv1\\Desktop\\Task1')
model = joblib.load('trained_model.pkl')

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/estimate', methods=['POST'])
def estimate():
    p_type = request.form['type']
    index_value = float(request.form['City'])
    floor = int(request.form['floor'])
    area = float(request.form['Area'])
    has_elevator = bool(request.form.get('has_elevator'))
    has_parking = bool(request.form.get('has_parking'))
    has_bars = bool(request.form.get('has_bars'))
    has_storage = bool(request.form.get('has_storage'))
    condition = request.form['condition']
    has_air_conditioner = bool(request.form.get('has_air_conditioner'))
    has_balcony = bool(request.form.get('has_balcony'))
    furniture = request.form['furniture']
    handicap_friendly = bool(request.form.get('handicap_friendly'))

    input_data = [
        area,
        has_elevator,
        has_parking,
        has_bars,
        has_storage,
        has_air_conditioner,
        has_balcony,
        handicap_friendly,
        floor,
        index_value,
        p_type,
        condition,
        furniture
    ]

    column_names = ['Area', 'hasElevator', 'hasParking', 'hasBars', 'hasStorage',
                    'hasAirCondition', 'hasBalcony', 'handicapFriendly', 'floor',
                    'Index_value', 'type', 'condition', 'furniture']

    input_df = pd.DataFrame([input_data], columns=column_names)



    estimated_price = model.predict(input_df)



    return render_template('result.html', estimated_price=estimated_price)

if __name__ == '__main__':
    app.run(debug=True)
