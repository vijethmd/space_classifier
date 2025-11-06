import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow import keras
import plotly.graph_objects as go
import plotly.express as px

# Set page config
st.set_page_config(
    page_title="Multi-Model Space Classifier",
    page_icon="🌌",
    layout="wide"
)

@st.cache_resource
def load_models():
    """Load all trained models"""
    models = {}
    try:
        # Load preprocessing
        models['scaler'] = joblib.load('scaler.pkl')
        models['le'] = joblib.load('label_encoder.pkl')
        models['feature_names'] = joblib.load('feature_names.pkl')
        
        # Load ML models
        models['rf'] = joblib.load('random_forest_model.pkl')
        models['dnn'] = keras.models.load_model('dnn_model.h5')
        models['cnn'] = keras.models.load_model('cnn_model.h5')
        
        return models
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.info("Please run `python train_all_models.py` first to train all models.")
        return None

def predict_with_all_models(models, features):
    """Get predictions from all three models"""
    # Prepare features
    features_df = pd.DataFrame([features], columns=models['feature_names'])
    features_scaled = models['scaler'].transform(features_df)
    features_cnn = features_scaled.reshape(features_scaled.shape[0], features_scaled.shape[1], 1)
    
    predictions = {}
    
    # Random Forest prediction
    rf_pred_encoded = models['rf'].predict(features_scaled)[0]
    rf_proba = models['rf'].predict_proba(features_scaled)[0]
    predictions['rf'] = {
        'class': models['le'].inverse_transform([rf_pred_encoded])[0],
        'confidence': np.max(rf_proba),
        'probabilities': rf_proba
    }
    
    # Deep Neural Network prediction
    dnn_proba = models['dnn'].predict(features_scaled)[0]
    dnn_pred_encoded = np.argmax(dnn_proba)
    predictions['dnn'] = {
        'class': models['le'].inverse_transform([dnn_pred_encoded])[0],
        'confidence': np.max(dnn_proba),
        'probabilities': dnn_proba
    }
    
    # CNN prediction
    cnn_proba = models['cnn'].predict(features_cnn)[0]
    cnn_pred_encoded = np.argmax(cnn_proba)
    predictions['cnn'] = {
        'class': models['le'].inverse_transform([cnn_pred_encoded])[0],
        'confidence': np.max(cnn_proba),
        'probabilities': cnn_proba
    }
    
    return predictions

def create_comparison_chart(predictions, le):
    """Create a comparison chart of all model predictions"""
    models = ['Random Forest', 'Deep Neural Network', '1D CNN']
    classes = le.classes_
    
    # Create grouped bar chart
    fig = go.Figure()
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # Different colors for each model
    
    for i, (model_key, color) in enumerate(zip(['rf', 'dnn', 'cnn'], colors)):
        model_name = models[i]
        probabilities = predictions[model_key]['probabilities']
        
        fig.add_trace(go.Bar(
            name=model_name,
            x=classes,
            y=probabilities,
            marker_color=color,
            text=[f'{p:.3f}' for p in probabilities],
            textposition='auto',
        ))
    
    fig.update_layout(
        title='Model Probability Comparison',
        xaxis_title='Class',
        yaxis_title='Probability',
        barmode='group',
        height=400,
        showlegend=True
    )
    
    return fig

def create_confidence_gauge(predictions):
    """Create confidence gauges for all models"""
    models = ['Random Forest', 'Deep Neural Network', '1D CNN']
    model_keys = ['rf', 'dnn', 'cnn']
    
    fig = go.Figure()
    
    for i, (model_name, model_key) in enumerate(zip(models, model_keys)):
        confidence = predictions[model_key]['confidence']
        
        fig.add_trace(go.Indicator(
            mode = "gauge+number+delta",
            value = confidence * 100,
            domain = {'row': i, 'column': 0},
            title = {'text': model_name},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "gray"},
                    {'range': [80, 100], 'color': "lightblue"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
    
    fig.update_layout(
        grid = {'rows': 3, 'columns': 1, 'pattern': "independent"},
        height=600
    )
    
    return fig

def main():
    st.title("🌌 Multi-Model Space Object Classifier")
    st.markdown("""
    Compare predictions from **three different AI models** for classifying astronomical objects!
    Choose your input features and see how each model performs.
    """)
    
    # Load models
    models = load_models()
    if models is None:
        return
    
    # Sidebar for model selection and input
    st.sidebar.header("🔧 Configuration")
    
    st.sidebar.subheader("🤖 Select Models to Compare")
    use_rf = st.sidebar.checkbox("Random Forest", value=True, help="Ensemble of decision trees")
    use_dnn = st.sidebar.checkbox("Deep Neural Network", value=True, help="Multi-layer neural network")
    use_cnn = st.sidebar.checkbox("1D CNN", value=True, help="Convolutional Neural Network for patterns")
    
    st.sidebar.markdown("---")
    st.sidebar.header("🔭 Input Features")
    
    # Input sliders
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        u_mag = st.slider("u magnitude", 14.0, 25.0, 18.3, 0.1)
        g_mag = st.slider("g magnitude", 14.0, 25.0, 17.8, 0.1)
        r_mag = st.slider("r magnitude", 14.0, 25.0, 17.5, 0.1)
    
    with col2:
        i_mag = st.slider("i magnitude", 14.0, 25.0, 17.2, 0.1)
        z_mag = st.slider("z magnitude", 14.0, 25.0, 17.0, 0.1)
        redshift = st.slider("Redshift", 0.0, 5.0, 0.1, 0.01)
    
    # Prepare features with engineering
    basic_features = [u_mag, g_mag, r_mag, i_mag, z_mag, redshift]
    engineered_features = [
        u_mag - g_mag,  # u-g color
        g_mag - r_mag,  # g-r color  
        r_mag - i_mag,  # r-i color
        i_mag - z_mag   # i-z color
    ]
    all_features = basic_features + engineered_features
    
    # Model information
    with st.sidebar.expander("ℹ️ About the Models"):
        st.markdown("""
        **🌲 Random Forest**
        - Ensemble of decision trees
        - Great for tabular data
        - Fast predictions
        
        **🧠 Deep Neural Network**  
        - Multi-layer perceptron
        - Learns complex patterns
        - Good generalization
        
        **📷 1D CNN**
        - Convolutional Neural Network
        - Excellent for pattern recognition
        - Uses 1D convolutions on features
        """)
    
    # Classification button
    if st.sidebar.button("🚀 Compare All Models", type="primary", use_container_width=True):
        # Get predictions from selected models
        selected_predictions = {}
        model_info = {
            'rf': ('🌲 Random Forest', use_rf),
            'dnn': ('🧠 Deep Neural Network', use_dnn), 
            'cnn': ('📷 1D CNN', use_cnn)
        }
        
        all_predictions = predict_with_all_models(models, all_features)
        
        for model_key, (model_name, is_selected) in model_info.items():
            if is_selected:
                selected_predictions[model_key] = all_predictions[model_key]
        
        if not selected_predictions:
            st.warning("Please select at least one model to compare!")
            return
        
        # Display results
        st.header("🎯 Model Comparison Results")
        
        # Results in columns
        cols = st.columns(len(selected_predictions))
        
        emoji_map = {"GALAXY": "🌌", "STAR": "⭐", "QSO": "⚡"}
        
        for col, (model_key, prediction) in zip(cols, selected_predictions.items()):
            model_name = model_info[model_key][0]
            emoji = emoji_map.get(prediction['class'], "🔭")
            
            with col:
                # Confidence color
                confidence = prediction['confidence']
                if confidence > 0.8:
                    confidence_color = "🟢"
                elif confidence > 0.6:
                    confidence_color = "🟡" 
                else:
                    confidence_color = "🟠"
                
                st.subheader(model_name)
                st.metric("Prediction", f"{emoji} {prediction['class']}")
                st.metric("Confidence", f"{confidence_color} {confidence:.2%}")
                
                # Mini probability chart
                prob_df = pd.DataFrame({
                    'Class': models['le'].classes_,
                    'Probability': prediction['probabilities']
                })
                st.bar_chart(prob_df.set_index('Class'), height=200)
        
        # Comparison visualizations
        st.header("📊 Detailed Comparison")
        
        tab1, tab2, tab3 = st.tabs(["Probability Comparison", "Confidence Gauges", "Model Insights"])
        
        with tab1:
            st.plotly_chart(create_comparison_chart(all_predictions, models['le']), use_container_width=True)
        
        with tab2:
            st.plotly_chart(create_confidence_gauge(all_predictions), use_container_width=True)
        
        with tab3:
            # Agreement analysis
            predictions_list = [pred['class'] for pred in all_predictions.values()]
            agreement = len(set(predictions_list)) == 1  # All models agree
            
            if agreement:
                st.success("✅ All models agree on the classification!")
            else:
                st.warning("⚠️ Models disagree on the classification")
                st.write("Different models may pick up on different patterns in the data")
            
            # Model strengths
            st.subheader("💡 Model Characteristics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**🌲 Random Forest**")
                st.write("• Robust to outliers")
                st.write("• Fast training")
                st.write("• Good for tabular data")
            
            with col2:
                st.write("**🧠 Deep Neural Network**")
                st.write("• Learns complex patterns")  
                st.write("• Good generalization")
                st.write("• Scalable to large data")
            
            with col3:
                st.write("**📷 1D CNN**")
                st.write("• Excellent pattern recognition")
                st.write("• Handles local dependencies")
                st.write("• Good for sequential data")
        
        # Feature analysis
        st.header("🔍 Feature Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Input Values")
            feature_display = [
                ("u magnitude", u_mag),
                ("g magnitude", g_mag), 
                ("r magnitude", r_mag),
                ("i magnitude", i_mag),
                ("z magnitude", z_mag),
                ("Redshift", redshift)
            ]
            
            for name, value in feature_display:
                st.write(f"**{name}:** {value:.2f}")
        
        with col2:
            st.subheader("Color Indices")
            colors = [
                ("u-g", u_mag - g_mag),
                ("g-r", g_mag - r_mag),
                ("r-i", r_mag - i_mag), 
                ("i-z", i_mag - z_mag)
            ]
            
            for color_name, value in colors:
                color_desc = "🔵 Blue" if value < 0 else "🔴 Red" if value > 0.8 else "🟡 Intermediate"
                st.write(f"**{color_name}:** {value:.2f} ({color_desc})")

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **Multi-Model Classifier**
    - Compare different AI approaches
    - See model agreement
    - Understand model strengths
    """)

if __name__ == "__main__":
    main()