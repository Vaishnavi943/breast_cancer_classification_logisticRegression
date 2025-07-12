import pickle
import numpy as np


# Load the trained model
loaded_model = pickle.load(open("D:/Projects-stuffs/Datasets-py/breast_cancer_model.sav", "rb"))

input_data = (7.76,	24.54,	47.92,	181.0,	0.05263	,0.04362	,0.00000,	0.00000,	0.1587,	0.05884,	0.3857,	1.428	,2.548,	19.15,	0.007189,	0.00466,	0.00000,	0.00000,	0.02676,	0.002783,	9.456	,30.37,	59.16	,268.6,	0.08996,	0.06444	,0.0000	,0.0000,	0.2871,	0.07039)
# change input data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array as we are predicting for one datapoint
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# prediction

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The Breast cancer is Malignant')

else:
  print('The Breast Cancer is Benign')