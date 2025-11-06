import streamlit as st
import numpy as np
import pandas as pd
import joblib
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
    """Load all available scikit-learn models"""
    models = {
        'rf': {'name': '🌲 Random Forest', 'model': None, 'color': '#2E86AB'},
        'gb': {'name': '🎯 Gradient Boosting', 'model': None, 'color': '#A23B72'}, 
        'svm': {'name': '⚡ Support Vector Machine', 'model': None, 'color': '#F18F01'},
        'knn': {'name': '📊 K-Nearest Neighbors', 'model': None, 'color': '#C73E1D'},
    }
    
    available_models = []
    
    try:
        # Load preprocessing
        models['scaler'] = joblib.load('scaler.pkl')
        models['le'] = joblib.load('label_encoder.pkl')
        models['feature_names'] = joblib.load('feature_names.pkl')
        available_models.append("Preprocessing")
    except:
        st.error("❌ Preprocessing files not found. Please train models first.")
        return None
    
    # Try to load each model
    for model_key in ['rf', 'gb', 'svm', 'knn']:
        try:
            models[model_key]['model'] = joblib.load(f'{model_key}_model.pkl')
            available_models.append(models[model_key]['name'])
        except:
            st.info(f"ℹ️ {models[model_key]['name']} not available")
    
    if available_models:
        st.success(f"✅ Loaded: {', '.join(available_models)}")
    
    return models

def predict_with_models(models, features):
    """Get predictions from all available models"""
    predictions = {}
    
    # Prepare features
    features_df = pd.DataFrame([features], columns=models['feature_names'])
    features_scaled = models['scaler'].transform(features_df)
    
    for model_key in ['rf', 'gb', 'svm', 'knn']:
        if models[model_key]['model'] is not None:
            model = models[model_key]['model']
            model_name = models[model_key]['name']
            
            try:
                pred_encoded = model.predict(features_scaled)[0]
                
                # Get probabilities if available
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(features_scaled)[0]
                    confidence = np.max(probabilities)
                else:
                    # For models without probability, use 1.0 for the predicted class
                    probabilities = np.zeros(len(models['le'].classes_))
                    probabilities[pred_encoded] = 1.0
                    confidence = 1.0
                
                predictions[model_key] = {
                    'name': model_name,
                    'class': models['le'].inverse_transform([pred_encoded])[0],
                    'confidence': confidence,
                    'probabilities': probabilities,
                    'color': models[model_key]['color']
                }
            except Exception as e:
                st.error(f"Error with {model_name}: {e}")
    
    return predictions

def create_comparison_plot(predictions, class_names):
    """Create a beautiful comparison plot"""
    fig = go.Figure()
    
    for model_key, pred_info in predictions.items():
        fig.add_trace(go.Bar(
            name=pred_info['name'],
            x=class_names,
            y=pred_info['probabilities'],
            marker_color=pred_info['color'],
            text=[f'{p:.3f}' for p in pred_info['probabilities']],
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

def create_confidence_radar(predictions):
    """Create a radar chart of model confidences"""
    model_names = [pred['name'] for pred in predictions.values()]
    confidences = [pred['confidence'] for pred in predictions.values()]
    colors = [pred['color'] for pred in predictions.values()]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=confidences + [confidences[0]],  # Close the circle
        theta=model_names + [model_names[0]],
        fill='toself',
        fillcolor='rgba(46, 134, 171, 0.3)',
        line=dict(color='#2E86AB'),
        name='Model Confidence'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=False,
        height=400,
        title='Model Confidence Comparison'
    )
    
    return fig

def main():
    st.title("🌌 Multi-Model Space Classifier")
    st.markdown("""
    Compare predictions from **multiple machine learning models**! 
    No TensorFlow required - fast, reliable, and always works! 🚀
    """)
    
    # Load models
    models = load_models()
    if models is None:
        st.stop()
    
    # Check available models
    available_models = [k for k in ['rf', 'gb', 'svm', 'knn'] if models[k]['model'] is not None]
    if not available_models:
        st.error("❌ No models found. Please train models first.")
        st.info("Run: `python train_no_tensorflow.py`")
        st.stop()
    
    # Sidebar
    st.sidebar.header("🔧 Configuration")
    
    # Model selection
    st.sidebar.subheader("🤖 Select Models to Compare")
    selected_models = {}
    for model_key in available_models:
        selected_models[model_key] = st.sidebar.checkbox(
            models[model_key]['name'], 
            value=True,
            help=f"Use {models[model_key]['name']} for prediction"
        )
    
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
    
    # Prepare features
    basic_features = [u_mag, g_mag, r_mag, i_mag, z_mag, redshift]
    engineered_features = [
        u_mag - g_mag, g_mag - r_mag, r_mag - i_mag, i_mag - z_mag
    ]
    all_features = basic_features + engineered_features
    
    # Model info
    with st.sidebar.expander("ℹ️ About the Models"):
        st.markdown("""
        **🌲 Random Forest**
        - Ensemble of decision trees
        - Robust and accurate
        
        **🎯 Gradient Boosting**
        - Builds trees sequentially
        - Often very accurate
        
        **⚡ Support Vector Machine**  
        - Finds optimal boundaries
        - Good for complex patterns
        
        **📊 K-Nearest Neighbors**
        - Based on similar examples
        - Simple but effective
        """)
    
    # Classification button
    if st.sidebar.button("🚀 Compare All Models", type="primary", use_container_width=True):
        # Get predictions
        all_predictions = predict_with_models(models, all_features)
        selected_predictions = {k: v for k, v in all_predictions.items() 
                              if k in selected_models and selected_models[k]}
        
        if not selected_predictions:
            st.warning("Please select at least one model!")
            return
        
        # Display results
        st.header("🎯 Model Predictions")
        
        # Results in columns
        cols = st.columns(len(selected_predictions))
        
        emoji_map = {"GALAXY": "🌌", "STAR": "⭐", "QSO": "⚡"}
        
        for col, (model_key, prediction) in zip(cols, selected_predictions.items()):
            with col:
                emoji = emoji_map.get(prediction['class'], "🔭")
                confidence = prediction['confidence']
                
                # Confidence indicator
                if confidence > 0.8:
                    confidence_emoji = "🟢"
                    confidence_text = "High"
                elif confidence > 0.6:
                    confidence_emoji = "🟡" 
                    confidence_text = "Medium"
                else:
                    confidence_emoji = "🟠"
                    confidence_text = "Low"
                
                st.subheader(prediction['name'])
                st.metric("Prediction", f"{emoji} {prediction['class']}")
                st.metric("Confidence", f"{confidence_emoji} {confidence:.2%}")
                st.caption(f"{confidence_text} confidence")
                
                # Mini probability chart
                prob_df = pd.DataFrame({
                    'Class': models['le'].classes_,
                    'Probability': prediction['probabilities']
                })
                st.bar_chart(prob_df.set_index('Class'), height=200)
        
        # Visualizations
        st.header("📊 Model Comparison")
        
        tab1, tab2 = st.tabs(["Probability Comparison", "Confidence Radar"])
        
        with tab1:
            st.plotly_chart(create_comparison_plot(all_predictions, models['le'].classes_), 
                          use_container_width=True)
        
        with tab2:
            st.plotly_chart(create_confidence_radar(all_predictions), 
                          use_container_width=True)
        
        # Agreement analysis
        st.header("🤝 Model Agreement")
        predictions_list = [pred['class'] for pred in selected_predictions.values()]
        unique_predictions = set(predictions_list)
        
        if len(unique_predictions) == 1:
            st.success(f"✅ All {len(selected_predictions)} models agree: **{list(unique_predictions)[0]}**")
        else:
            st.warning(f"⚠️ Models disagree: {', '.join(unique_predictions)}")
            
            # Show which models predicted what
            agreement_df = pd.DataFrame([
                {'Model': pred['name'], 'Prediction': pred['class'], 'Confidence': pred['confidence']}
                for pred in selected_predictions.values()
            ])
            st.dataframe(agreement_df, use_container_width=True)
        
        # Feature analysis
        st.header("🔍 Feature Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Input Values")
            for feature, value in zip(['u', 'g', 'r', 'i', 'z', 'redshift'], basic_features):
                st.write(f"**{feature}:** {value:.2f}")
        
        with col2:
            st.subheader("Color Indices")
            colors = [
                ("u-g", u_mag - g_mag),
                ("g-r", g_mag - r_mag), 
                ("r-i", r_mag - i_mag),
                ("i-z", i_mag - z_mag)
            ]
            for color_name, value in colors:
                color_type = "🔵 Blue" if value < 0 else "🔴 Red" if value > 0.8 else "🟡 Intermediate"
                st.write(f"**{color_name}:** {value:.2f} ({color_type})")

if __name__ == "__main__":
    main()