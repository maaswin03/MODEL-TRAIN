import pandas as pd
import joblib

data = pd.read_csv('./datsets/rainfall in india 1901-2015.csv')

data.fillna(0, inplace=True)

def get_rainfall_data(state_name):
    filtered_data = data[data['SUBDIVISION'] == state_name]

    average_monthly_rainfall = filtered_data[['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']].mean()

    avg_for_next_three_months = average_monthly_rainfall.iloc[0:3].mean()

    return average_monthly_rainfall, avg_for_next_three_months

state_name = "TAMILNADU"
monthly_avg, three_month_avg = get_rainfall_data(state_name)
print(f"Average Monthly Rainfall for {state_name}:\n{monthly_avg}")
print(f"\nAverage Rainfall for the next three months: {three_month_avg:.2f}")

joblib.dump(get_rainfall_data, 'rainfall_prediction_function.pkl')
joblib.dump(data, 'rainfall_data.pkl')

