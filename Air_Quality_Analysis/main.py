import streamlit as st
import joblib
import pickle

logistic_model = joblib.load('logistic_model.pkl')
svm_model = joblib.load('svm_model.pkl')

# Function to predict air quality
def predict_air_quality(model, data):
    prediction = model.predict(data)
    return prediction

# Mapping for predicted values to air quality categories
air_quality_mapping = {
    1: 'Moderate',
    2: 'Poor',
    5: 'Very Poor',
    4: 'Severe',
    3: 'Satisfactory',
    0: 'Good'
}

# Main function to run the web app
def main():
    st.title('Air Quality Prediction')

    st.write('Enter the following information:')
    location = st.text_input('Location (Area Name):')

    # Input fields
    pm25 = st.number_input('PM2.5')
    no = st.number_input('NO')
    no2 = st.number_input('NO2')
    nox = st.number_input('NOx')
    co = st.number_input('CO')
    so2 = st.number_input('SO2')
    o3 = st.number_input('O3')
    benzene = st.number_input('Benzene')
    aqi = st.number_input('AQI')

    # Buttons to select models
    if st.button('Model 1'):
        data = [[pm25, no, no2, nox, co, so2, o3, benzene, aqi]]
        prediction = predict_air_quality(logistic_model, data)
        st.write(f'Air Quality Prediction for {location}: {air_quality_mapping[prediction[0]]}')

    if st.button('Model 2'):
        data = [[pm25, no, no2, nox, co, so2, o3, benzene, aqi]]
        prediction = predict_air_quality(svm_model, data)
        st.write(f'Air Quality Prediction for {location}: {air_quality_mapping[prediction[0]]}')

if __name__ == "__main__":
    main()
