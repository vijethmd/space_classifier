import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Set page config
st.set_page_config(
    page_title="SDSS Space Classifier Pro",
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
        st.error("Models not found! Please train the improved model first.")
        return None, None, None, None

def main():
    st.title("🌌 SDSS Space Object Classifier Pro")
    st.markdown("""
    Classify astronomical objects from the Sloan Digital Sky Survey as **Galaxies**, **Stars**, or **Quasars**.
    Uses improved machine learning with engineered features for better accuracy!
    """)
    
    # Load models
    rf_model, scaler, le, feature_names = load_models()
    
    if rf_model is None:
        st.info("""
        To train the improved model, run in terminal:
        ```bash
        python train_improved.py
        ```
        """)
        return
    
    # Sidebar for input
    st.sidebar.header("🔭 Astronomical Measurements")
    
    # Main magnitude inputs
    st.sidebar.subheader("📊 Photometric Magnitudes")
    u_mag = st.sidebar.slider("u magnitude", 14.0, 25.0, 18.3, 0.1, 
                             help="Ultraviolet magnitude")
    g_mag = st.sidebar.slider("g magnitude", 14.0, 25.0, 17.8, 0.1,
                             help="Green magnitude") 
    r_mag = st.sidebar.slider("r magnitude", 14.0, 25.0, 17.5, 0.1,
                             help="Red magnitude")
    i_mag = st.sidebar.slider("i magnitude", 14.0, 25.0, 17.2, 0.1,
                             help="Near-infrared magnitude")
    z_mag = st.sidebar.slider("z magnitude", 14.0, 25.0, 17.0, 0.1,
                             help="Infrared magnitude")
    
    st.sidebar.subheader("🌐 Cosmological Features")
    redshift = st.sidebar.slider("Redshift", 0.0, 5.0, 0.1, 0.01,
                                help="Cosmological redshift - indicates distance")
    
    # Calculate color indices (important for classification)
    u_g_color = u_mag - g_mag
    g_r_color = g_mag - r_mag
    r_i_color = r_mag - i_mag
    i_z_color = i_mag - z_mag
    
    # Prepare features in correct order
    basic_features = [u_mag, g_mag, r_mag, i_mag, z_mag, redshift]
    engineered_features = [u_g_color, g_r_color, r_i_color, i_z_color]
    all_features = basic_features + engineered_features
    
    if st.sidebar.button("🚀 Classify Object", type="primary"):
        # Create feature DataFrame
        features_df = pd.DataFrame([all_features], columns=feature_names)
        features_scaled = scaler.transform(features_df)
        
        # Make prediction
        prediction_encoded = rf_model.predict(features_scaled)[0]
        probabilities = rf_model.predict_proba(features_scaled)[0]
        confidence = np.max(probabilities)
        prediction = le.inverse_transform([prediction_encoded])[0]
        
        # Display results
        st.header("🎯 Classification Results")
        
        # Results in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Prediction with emoji
            emoji_map = {"GALAXY": "🌌", "STAR": "⭐", "QSO": "⚡"}
            emoji = emoji_map.get(prediction, "🔭")
            st.metric("Prediction", f"{emoji} {prediction}")
            
            # Confidence with color
            confidence_color = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.6 else "🟠"
            st.metric("Confidence", f"{confidence_color} {confidence:.2%}")
        
        with col2:
            # Probability chart
            st.subheader("Class Probabilities")
            prob_df = pd.DataFrame({
                'Class': le.classes_,
                'Probability': probabilities
            })
            # Add emojis to class names
            prob_df['Class'] = prob_df['Class'].map({
                'GALAXY': '🌌 GALAXY',
                'STAR': '⭐ STAR', 
                'QSO': '⚡ QSO'
            })
            st.bar_chart(prob_df.set_index('Class'))
        
        with col3:
            # Confidence gauge
            st.subheader("Confidence Level")
            if confidence > 0.8:
                st.success("🔍 High Confidence")
                st.write("The model is very certain about this classification")
            elif confidence > 0.6:
                st.warning("💡 Medium Confidence") 
                st.write("The model is somewhat certain")
            else:
                st.error("❓ Low Confidence")
                st.write("The model is uncertain - consider this a tentative classification")
        
        # Feature analysis
        st.header("📊 Feature Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Input Measurements")
            measurements = [
                ("u magnitude", u_mag, "Ultraviolet brightness"),
                ("g magnitude", g_mag, "Green light brightness"), 
                ("r magnitude", r_mag, "Red light brightness"),
                ("i magnitude", i_mag, "Near-infrared brightness"),
                ("z magnitude", z_mag, "Infrared brightness"),
                ("Redshift", redshift, "Cosmological distance indicator")
            ]
            
            for name, value, desc in measurements:
                st.write(f"**{name}:** {value:.2f}")
                st.caption(f"{desc}")
        
        with col2:
            st.subheader("Color Indices")
            st.write("Color indices help distinguish object types:")
            
            colors = [
                ("u-g", u_g_color, "UV vs Green - hot objects are negative"),
                ("g-r", g_r_color, "Green vs Red - stars vs galaxies"), 
                ("r-i", r_i_color, "Red vs IR - temperature indicator"),
                ("i-z", i_z_color, "IR colors - dust absorption")
            ]
            
            for color_name, value, interpretation in colors:
                color_emoji = "🔵" if value < 0 else "🔴" if value > 1 else "🟡"
                st.write(f"**{color_name}:** {color_emoji} {value:.2f}")
                st.caption(interpretation)
            
            # Astronomical interpretation
            st.subheader("🔭 Astronomical Insights")
            insights = []
            
            if redshift > 1.0:
                insights.append("🚀 High redshift - very distant object!")
            elif redshift < 0.01:
                insights.append("🌍 Low redshift - relatively nearby object")
                
            if u_g_color < 0:
                insights.append("🔵 Very blue - likely hot star or quasar")
            elif g_r_color > 0.8:
                insights.append("🔴 Very red - likely cool star or dusty galaxy")
                
            for insight in insights:
                st.write(insight)

        # Object type information
        st.header("📚 About the Classes")
        
        class_info = {
            "GALAXY": "A massive system of stars, gas, and dark matter bound by gravity. Our Milky Way is a galaxy.",
            "STAR": "A luminous sphere of plasma held together by gravity, like our Sun.", 
            "QSO": "Quasi-Stellar Object - extremely luminous active galactic nuclei, among the brightest objects in the universe."
        }
        
        for class_name, description in class_info.items():
            with st.expander(f"{emoji_map.get(class_name, '🔭')} {class_name}"):
                st.write(description)

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **About this classifier:**
    - Uses Random Forest machine learning
    - Trained on SDSS data with engineered features
    - Color indices improve classification accuracy
    """)

if __name__ == "__main__":
    main()