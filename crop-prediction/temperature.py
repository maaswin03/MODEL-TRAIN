import pandas as pd
import joblib

data = pd.read_csv('./datasets/temperature_data_india_2000_2023.csv')

data.fillna(0, inplace=True)

def get_temperature_data(state_name):
    filtered_data = data[data['SUBDIVISION'] == state_name]

    average_monthly_temperature = filtered_data[['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']].mean()

    avg_for_next_three_months = average_monthly_temperature.iloc[0:3].mean()

    return average_monthly_temperature, avg_for_next_three_months

state_name = "TAMIL NADU"
monthly_avg, three_month_avg = get_temperature_data(state_name)
print(f"Average Monthly temperature for {state_name}:\n{monthly_avg}")
print(f"\nAverage temperature for the next three months: {three_month_avg:.2f}")

joblib.dump(get_temperature_data, 'temperature_prediction_function.pkl')
joblib.dump(data, 'temperature_data.pkl')
