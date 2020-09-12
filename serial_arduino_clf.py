import serial  # to handel serial data to Arduino
import tensorflow as tf
import numpy as np
from time import sleep  # for delay

comm_port = '/dev/cu.usbmodem14101'  # Arduino communication port
baud_rate = 9600  # Arduino comm. baud
impedance = []  # empty list to store impedance data

gf_arr = [0.1565, 1.1504, 1.8395, 2.3795, 2.6402, 2.6264, 2.6654, 2.73407, 2.7488, 2.7378,
          2.7430, 2.8000, 2.8278, 2.8195, 2.8071, 2.8325, 2.8378, 2.8503, 2.8777, 2.8563,
          2.8623, 2.8606, 2.8684, 2.8783, 2.8909, 2.8825, 2.8757, 2.8842, 2.8886, 2.8975,
          2.9054, 2.9050, 2.9153, 2.9176, 2.9242, 2.9284, 2.9409, 2.9429, 2.9424, 2.9478,
          2.9528, 2.9604, 2.9626, 2.9720, 2.9701, 2.9715, 2.9777, 2.9818, 2.9962, 3.0031,
          2.9988, 3.0082, 3.0152, 3.0166, 3.0215, 3.0305, 3.0367, 3.0376, 3.0430, 3.0497,
          3.0499, 3.0552, 3.0658, 3.0701, 3.0724, 3.0817, 3.0878, 3.0904, 3.0984, 3.1040,
          3.1139, 3.1130, 3.1088, 3.1082, 3.1123, 3.1213, 3.1289, 3.1319, 3.1352, 3.1397,
          3.1457, 3.1533, 3.1627, 3.1706, 3.1782, 3.1838, 3.1864, 3.1915, 3.1951, 3.2030,
          3.2117, 3.2193, 3.2285, 3.2327, 3.2398, 3.2466, 3.2524, 3.2592, 3.2700, 3.2756]

ser = serial.Serial(comm_port, baud_rate, timeout=3.0)
sleep(2)
ser.write('C'.encode())
sleep(2)
for itr in range(100):
    data_read = ser.readline()
    impedance.append(float(1/(gf_arr[itr]*pow(10, -10)*float(data_read))))
ser.close()

print(impedance)

x_test = np.array(impedance, dtype='float32')

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="saved_model/tflite_model/converted_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

test = np.vstack([x_test])
interpreter.set_tensor(input_details[0]['index'], test)

interpreter.invoke()

classes = interpreter.get_tensor(output_details[0]['index'])
if classes[0] > 0.5:
    print("is Cancerous")
else:
    print("is not Cancerous")