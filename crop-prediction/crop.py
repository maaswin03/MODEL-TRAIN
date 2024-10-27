import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

data = pd.read_csv('./datasets/Crop_recommendation.csv')

print(data.head())
print(data.info())
print(data['label'].value_counts())

X = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = data['label']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

joblib.dump(model, 'crop_recommendation_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

loaded_model = joblib.load('crop_recommendation_model.pkl')
loaded_scaler = joblib.load('scaler.pkl')

input_features = [20.5, 65.0, 60.5, 25, 80, 35, 500]
input_features_scaled = loaded_scaler.transform([input_features])
prediction = loaded_model.predict(input_features_scaled)
print("Predicted crop:", prediction[0])
