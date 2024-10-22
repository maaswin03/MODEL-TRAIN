import numpy as np
import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="flood_prediction_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_data = np.array([[575, 830, 760], [5, 10, 60], [50, 300, 205]], dtype=np.float32)

for i in range(input_data.shape[0]):
    interpreter.set_tensor(input_details[0]['index'], [input_data[i]])

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction = (output_data < 0.5).astype(int)
    
    if prediction == 1:
        print(f"Data point {i+1}: Possibility of severe flood")
    else:
        print(f"Data point {i+1}: No chance of severe flood")
