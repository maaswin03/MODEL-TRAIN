import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

x = pd.read_csv("./tamilnadu.csv")

y1 = list(x["YEAR"])
x1 = list(x["Oct-Dec"])
z1 = list(x["OCT"])
w1 = list(x["NOV"])

flood = []
Nov = []
sub = []

for i in range(len(x1)):
    if x1[i] > 500:
        flood.append(1)
    else:
        flood.append(0)
        
for k in range(len(x1)):
    Nov.append(z1[k] / 3)
    sub.append(abs(w1[k] - z1[k]))
    
x["flood"] = flood
x["avgnov"] = Nov
x["sub"] = sub

x.to_csv("out.csv", index=False)
print(x.head())

X = x[['Oct-Dec', 'OCT', 'NOV']].values
y = np.array(flood)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

y_pred = (model.predict(X_test) > 0.5).astype(int)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

new_data = np.array([[575, 830, 760], [0, 0, 0], [50, 300, 205]])
flood_predictions = (model.predict(new_data) < 0.5).astype(int)

for i, prediction in enumerate(flood_predictions):
    if prediction == 1:
        print(f"Data point {i+1}: Possibility of severe flood")
    else:
        print(f"Data point {i+1}: No chance of severe flood")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("flood_prediction_model.tflite", "wb") as f:
    f.write(tflite_model)

print("Model successfully converted to TensorFlow Lite!")
