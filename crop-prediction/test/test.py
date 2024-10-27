from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load('./crop_recommendation_model.pkl')
scaler = joblib.load('./scaler.pkl')

rainfall_data = joblib.load('./rainfall_data.pkl')

temperature_data = joblib.load('./temperature_data.pkl')

humidity_data = joblib.load('./humidity_data.pkl')


def get_rainfall_data(state_name):
    filtered_data = rainfall_data[rainfall_data['SUBDIVISION'] == state_name]

    average_monthly_rainfall = filtered_data[['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']].mean()

    return average_monthly_rainfall


def get_temperature_data(state_name):
    filtered_data = temperature_data[temperature_data['SUBDIVISION'] == state_name]

    average_monthly_temperature = filtered_data[['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']].mean()

    return average_monthly_temperature


def get_humidity_data(state_name):
    filtered_data = humidity_data[humidity_data['SUBDIVISION'] == state_name]

    average_monthly_humidity = filtered_data[['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']].mean()

    return average_monthly_humidity


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    features = [data['N'], data['P'], data['K'], data['temperature'], data['humidity'], data['ph'], data['rainfall']]
    
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)[0]

    state_name = data.get('state_name')
    current_month = data.get('current_month') 

    average_monthly_rainfall = get_rainfall_data(state_name)

    next_six_months_avg_rainfall = average_monthly_rainfall[current_month-1:current_month+5].mean()


    average_monthly_temperature = get_temperature_data(state_name)

    next_six_months_avg_temperature = average_monthly_temperature[current_month-1:current_month+5].mean()

    
    average_monthly_humidity = get_humidity_data(state_name)

    next_six_months_avg_humidity = average_monthly_humidity[current_month-1:current_month+5].mean()

    return jsonify({
        'crop': prediction,
        'average_rainfall_next_six_months': next_six_months_avg_rainfall,
        'average_temperature_next_six_months': next_six_months_avg_temperature,
        'average_humidity_next_six_months': next_six_months_avg_humidity
    })

if __name__ == '__main__':
    app.run(debug=True)
