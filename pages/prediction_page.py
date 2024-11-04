import streamlit as st
import requests
import pickle

# Cargar las características seleccionadas
features_path = '/Users/estebanjimenez/Library/CloudStorage/OneDrive-Personal/Tec_Monterrey/Maestria_Aya/MLOPS/ML_OPS_WINE/selected_features.pkl'

with open(features_path, "rb") as f:
    selected_features = pickle.load(f)

def prediction_page():
    st.title("Predicción")

    # Campos para las características seleccionadas
    features = []
    for feature_name in selected_features:
        feature_input = st.number_input(f'{feature_name}', value=0.0, step=0.1)
        features.append(feature_input)

    if st.button('Predecir'):
        try:
            response = requests.post(
                'http://localhost:8000/predict',
                json={'features': features}
            )
            if response.status_code == 200:
                prediction = response.json()
                st.success(f"El tipo de vino predicho es: {prediction['prediction']}")
            else:
                st.error(f'Error en la predicción: {response.json()["detail"]}')
        except requests.exceptions.RequestException as e:
            st.error(f"No se pudo conectar al servidor de predicción: {e}")
