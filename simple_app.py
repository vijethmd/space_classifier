import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Set page config
st.set_page_config(
    page_title="SDSS Space Classifier",
    page_icon="🌌",
    layout="wide"
)

@st.cache_resource
def load_models():
    try:
        rf_model = joblib.load('random_forest_model.pkl')
        scaler = joblib.load('scaler.pkl')
        le = joblib.load('label_encoder.pkl')
        feature_names = joblib.load('feature_names.pkl')
        return rf_model, scaler, le, feature_names
    except:
        st.error("Models not found! Please train the model first.")
        return None, None, None, None

def main():
    st.title("🌌 SDSS Space Object Classifier")
    
    # Load models
    rf_model, scaler, le, feature_names = load_models()
    
    if rf_model is None:
        st.info("""
        To train the model, run in terminal:
        ```bash
        python train_guaranteed.py
        ```
        """)
        return
    
    st.sidebar.header("📊 Input Features")
    
    # Dynamic input sliders based on available features
    features = []
    feature_values = {}
    
    # Common SDSS features with reasonable ranges
    feature_ranges = {
        'u': (14.0, 25.0, 19.0),
        'g': (14.0, 25.0, 18.5),
        'r': (14.0, 25.0, 17.5),
        'i': (14.0, 25.0, 17.0),
        'z': (14.0, 25.0, 16.8),
        'redshift': (0.0, 5.0, 0.1),
        'plate': (100, 8000, 2000),
        'MJD': (50000, 60000, 55000),
        'fiberID': (1, 1000, 500)
    }
    
    for feature in feature_names:
        if feature in feature_ranges:
            min_val, max_val, default_val = feature_ranges[feature]
            value = st.sidebar.slider(f"{feature}", min_val, max_val, default_val, 0.1)
            feature_values[feature] = value
            features.append(value)
        else:
            # For unknown features, use a reasonable default range
            value = st.sidebar.slider(f"{feature}", 0.0, 100.0, 50.0, 0.1)
            feature_values[feature] = value
            features.append(value)
    
    if st.sidebar.button("🔍 Classify Object"):
        # Prepare features for prediction
        features_df = pd.DataFrame([features], columns=feature_names)
        features_scaled = scaler.transform(features_df)
        
        # Make prediction
        prediction_encoded = rf_model.predict(features_scaled)[0]
        confidence = np.max(rf_model.predict_proba(features_scaled))
        prediction = le.inverse_transform([prediction_encoded])[0]
        
        st.header("🎯 Classification Result")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Prediction", prediction)
            st.metric("Confidence", f"{confidence:.2%}")
        
        with col2:
            # Show probabilities
            probabilities = rf_model.predict_proba(features_scaled)[0]
            prob_df = pd.DataFrame({
                'Class': le.classes_,
                'Probability': probabilities
            })
            st.bar_chart(prob_df.set_index('Class'))
        
        # Feature display
        st.header("📊 Input Features Used")
        for feature, value in feature_values.items():
            st.write(f"**{feature}:** {value}")

if __name__ == "__main__":
    main()