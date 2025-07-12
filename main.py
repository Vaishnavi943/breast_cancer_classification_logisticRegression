
import streamlit as st
import pickle
import numpy as np

# Load the trained model
loaded_model = pickle.load(open("D:/Projects-stuffs/Datasets-py/breast_cancer_model.sav", "rb"))

def breast_cancer_predict(input_data):

    # change input data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array as we are predicting for one datapoint
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    # prediction

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
        return 'The Breast cancer is Malignant, i.e; person has breast cancer.'

    else:
        return 'The Breast Cancer is Benign'



def main():
    # Title
    st.title("Breast Cancer Prediction App")
    st.markdown("Enter the patient features below:")

    # Define all input fields
    features = {
    "radius_mean": 0.0,
    "texture_mean": 0.0,
    "perimeter_mean": 0.0,
    "area_mean": 0.0,
    "smoothness_mean": 0.0,
    "compactness_mean": 0.0,
    "concavity_mean": 0.0,
    "concave points_mean": 0.0,
    "symmetry_mean": 0.0,
    "fractal_dimension_mean": 0.0,
    "radius_se": 0.0,
    "texture_se": 0.0,
    "perimeter_se": 0.0,
    "area_se": 0.0,
    "smoothness_se": 0.0,
    "compactness_se": 0.0,
    "concavity_se": 0.0,
    "concave points_se": 0.0,
    "symmetry_se": 0.0,
    "fractal_dimension_se": 0.0,
    "radius_worst": 0.0,
    "texture_worst": 0.0,
    "perimeter_worst": 0.0,
    "area_worst": 0.0,
    "smoothness_worst": 0.0,
    "compactness_worst": 0.0,
    "concavity_worst": 0.0,
    "concave points_worst": 0.0,
    "symmetry_worst": 0.0,
    "fractal_dimension_worst": 0.0
    }

    # Input collection
    for key in features:   
        features[key] = st.text_input(f"{key.replace('_', ' ').capitalize()}")

    diagnosis = ''
    # Prediction
    if st.button("Result"):

        # Convert all inputs to float before prediction as we have prepared model on the float value
        input_data = [float(value) for value in features.values()]

        diagnosis = breast_cancer_predict(input_data)

    st.success(diagnosis)


if __name__ == '__main__':
    main()


