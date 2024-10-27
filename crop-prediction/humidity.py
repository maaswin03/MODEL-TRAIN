import pandas as pd
import joblib

data = pd.read_csv('./datasets/humidity_data_india_2000_2023.csv')

data.fillna(0, inplace=True)

def get_humidity_data(state_name):
    filtered_data = data[data['SUBDIVISION'] == state_name]

    average_monthly_humidity = filtered_data[['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']].mean()

    avg_for_next_three_months = average_monthly_humidity.iloc[0:3].mean()

    return average_monthly_humidity, avg_for_next_three_months

state_name = "TAMIL NADU"
monthly_avg, three_month_avg = get_humidity_data(state_name)
print(f"Average Monthly humidity for {state_name}:\n{monthly_avg}")
print(f"\nAverage humidity for the next three months: {three_month_avg:.2f}")

joblib.dump(get_humidity_data, 'humidity_prediction_function.pkl')
joblib.dump(data, 'humidity_data.pkl')
