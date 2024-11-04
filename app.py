import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set the page config
st.set_page_config(page_title="Wine Classification App", layout="wide")

# Sidebar navigation
st.sidebar.title("Navegación")
page = st.sidebar.radio("Ir a:", ["Home", "Predicción", "Modelo"])

# Load the appropriate page
if page == "Home":
    from pages.home_page import home_page
    home_page()
elif page == "Predicción":
    from pages.prediction_page import prediction_page
    prediction_page()
elif page == "Modelo":
    from pages.model_page import model_page
    model_page()
