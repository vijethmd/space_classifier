import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow import keras

# Set page config
st.set_page_config(
    page_title="SDSS Space Object Classifier",
    page_icon="🌌",
    layout="wide"
)

# Load models (with caching)
@st.cache_resource
def load_models():
    try:
        rf_model = joblib.load('random_forest_model.pkl')
        dl_model = keras.models.load_model('deep_learning_model.h5')
        cnn_model = keras.models.load_model('cnn_model.h5')
        scaler = joblib.load('scaler.pkl')
        le = joblib.load('label_encoder.pkl')
        feature_names = joblib.load('feature_names.pkl')
        return rf_model, dl_model, cnn_model, scaler, le, feature_names
    except:
        st.error("Please train the models first by running the training scripts!")
        return None, None, None, None, None, None

def reshape_for_cnn(features, feature_names):
    """Reshape features for CNN input"""
    features_df = pd.DataFrame([features], columns=feature_names)
    features_scaled = scaler.transform(features_df)
    return features_scaled.reshape(features_scaled.shape[0], features_scaled.shape[1], 1)

def main():
    st.title("🌌 SDSS Space Object Classifier")
    st.markdown("""
    Identify whether an astronomical object from the **Sloan Digital Sky Survey (SDSS)** 
    is a **Galaxy**, **Star**, or **Quasar** using machine learning!
    
    **Dataset**: Stellar Classification Dataset - SDSS17 from Kaggle
    """)
    
    # Load models
    rf_model, dl_model, cnn_model, scaler, le, feature_names = load_models()
    
    if rf_model is None:
        return
    
    # Sidebar for input
    st.sidebar.header("📊 SDSS Input Features")
    
    st.sidebar.markdown("""
    **Feature Guide:**
    - **u, g, r, i, z**: Magnitudes in different filters
    - **Redshift**: Cosmological redshift (0-1 for nearby objects)
    - **Plate, MJD, fiberID**: Observation identifiers
    """)
    
    # Input sliders with realistic ranges from SDSS data
    u_mag = st.sidebar.slider("u magnitude", 14.0, 25.0, 19.0, 0.1)
    g_mag = st.sidebar.slider("g magnitude", 14.0, 25.0, 18.5, 0.1)
    r_mag = st.sidebar.slider("r magnitude", 14.0, 25.0, 17.5, 0.1)
    i_mag = st.sidebar.slider("i magnitude", 14.0, 25.0, 17.0, 0.1)
    z_mag = st.sidebar.slider("z magnitude", 14.0, 25.0, 16.8, 0.1)
    redshift = st.sidebar.slider("Redshift", 0.0, 5.0, 0.1, 0.01)
    plate = st.sidebar.slider("Plate", 100, 8000, 2000, 1)
    mjd = st.sidebar.slider("MJD", 50000, 60000, 55000, 1)
    fiberid = st.sidebar.slider("Fiber ID", 1, 1000, 500, 1)
    
    features = [u_mag, g_mag, r_mag, i_mag, z_mag, redshift, plate, mjd, fiberid]
    
    # Model selection
    st.sidebar.header("🤖 Model Selection")
    use_rf = st.sidebar.checkbox("Random Forest", value=True)
    use_dl = st.sidebar.checkbox("Deep Neural Network", value=True)
    use_cnn = st.sidebar.checkbox("1D CNN", value=True)
    
    # Make predictions when button is clicked
    if st.sidebar.button("🔍 Classify Object"):
        st.header("🎯 Classification Results")
        
        # Create columns for results
        col1, col2, col3 = st.columns(3)
        
        # Prepare features for prediction
        features_df = pd.DataFrame([features], columns=feature_names)
        features_scaled = scaler.transform(features_df)
        
        # Random Forest Prediction
        if use_rf:
            with col1:
                st.subheader("🌲 Random Forest")
                rf_pred_encoded = rf_model.predict(features_scaled)[0]
                rf_confidence = np.max(rf_model.predict_proba(features_scaled))
                rf_pred = le.inverse_transform([rf_pred_encoded])[0]
                
                st.metric("Prediction", rf_pred)
                st.metric("Confidence", f"{rf_confidence:.2%}")
                
                # Show probabilities
                rf_probs = rf_model.predict_proba(features_scaled)[0]
                prob_df = pd.DataFrame({
                    'Class': le.classes_,
                    'Probability': rf_probs
                })
                st.bar_chart(prob_df.set_index('Class'))
        
        # Deep Learning Prediction
        if use_dl:
            with col2:
                st.subheader("🧠 Deep Neural Network")
                dl_pred_proba = dl_model.predict(features_scaled)
                dl_pred_encoded = np.argmax(dl_pred_proba, axis=1)[0]
                dl_confidence = np.max(dl_pred_proba)
                dl_pred = le.inverse_transform([dl_pred_encoded])[0]
                
                st.metric("Prediction", dl_pred)
                st.metric("Confidence", f"{dl_confidence:.2%}")
                
                # Show probabilities
                dl_probs = dl_pred_proba[0]
                prob_df = pd.DataFrame({
                    'Class': le.classes_,
                    'Probability': dl_probs
                })
                st.bar_chart(prob_df.set_index('Class'))
        
        # CNN Prediction
        if use_cnn:
            with col3:
                st.subheader("📷 1D CNN")
                cnn_input = features_scaled.reshape(features_scaled.shape[0], features_scaled.shape[1], 1)
                cnn_pred_proba = cnn_model.predict(cnn_input)
                cnn_pred_encoded = np.argmax(cnn_pred_proba, axis=1)[0]
                cnn_confidence = np.max(cnn_pred_proba)
                cnn_pred = le.inverse_transform([cnn_pred_encoded])[0]
                
                st.metric("Prediction", cnn_pred)
                st.metric("Confidence", f"{cnn_confidence:.2%}")
                
                # Show probabilities
                cnn_probs = cnn_pred_proba[0]
                prob_df = pd.DataFrame({
                    'Class': le.classes_,
                    'Probability': cnn_probs
                })
                st.bar_chart(prob_df.set_index('Class'))
        
        # Feature analysis
        st.header("📊 Feature Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Input Features")
            feature_display_names = [
                'u magnitude', 'g magnitude', 'r magnitude', 'i magnitude', 'z magnitude',
                'Redshift', 'Plate', 'MJD', 'Fiber ID'
            ]
            for name, value in zip(feature_display_names, features):
                st.write(f"**{name}:** {value:.2f}")
        
        with col2:
            st.subheader("Astronomical Interpretation")
            interpretations = []
            
            # Color analysis (g - r)
            color_gr = g_mag - r_mag
            if color_gr < 0.3:
                interpretations.append("🔵 Blue object (hot star/young galaxy)")
            elif color_gr > 0.8:
                interpretations.append("🔴 Red object (cool star/old galaxy)")
            else:
                interpretations.append("🟡 Intermediate color")
            
            # Brightness analysis
            if r_mag < 16:
                interpretations.append("🔆 Very bright object")
            elif r_mag > 20:
                interpretations.append("🌑 Faint object")
            
            # Redshift analysis
            if redshift < 0.01:
                interpretations.append("🌍 Nearby object")
            elif redshift > 0.1:
                interpretations.append("🚀 Distant object")
            
            for interpretation in interpretations:
                st.write(interpretation)

    # Model information
    st.sidebar.header("ℹ️ About")
    st.sidebar.markdown("""
    **Models Used:**
    - 🌲 Random Forest: Ensemble of decision trees
    - 🧠 Deep Neural Network: Multi-layer perceptron  
    - 📷 1D CNN: Convolutional Neural Network for feature patterns
    
    **Classes:**
    - GALAXY: Collection of stars, gas, and dust
    - STAR: Luminous sphere of plasma
    - QSO: Quasi-stellar object (active galaxy nucleus)
    """)

if __name__ == "__main__":
    main()