import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt
from data_preparation import prepare_data

def train_random_forest(csv_file_path):
    """Train and evaluate a Random Forest classifier on SDSS data"""
    print("=== Training Random Forest Model ===")
    
    # Prepare data
    X_train, X_test, y_train, y_test, le, scaler, feature_names = prepare_data(csv_file_path)
    
    # Create and train the model
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1  # Use all available cores
    )
    
    print("Training Random Forest...")
    rf_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf_model.predict(X_test)
    y_pred_proba = rf_model.predict_proba(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"🎯 Random Forest Accuracy: {accuracy:.4f}")
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n🔝 Feature Importance:")
    print(feature_importance)
    
    # Plot feature importance (using matplotlib instead of seaborn)
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['feature'], feature_importance['importance'])
    plt.xlabel('Importance')
    plt.title('Random Forest Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Confusion matrix (using matplotlib)
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix - Random Forest')
    plt.colorbar()
    tick_marks = np.arange(len(le.classes_))
    plt.xticks(tick_marks, le.classes_, rotation=45)
    plt.yticks(tick_marks, le.classes_)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.savefig('confusion_matrix_rf.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save the model and preprocessing objects
    joblib.dump(rf_model, 'random_forest_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(le, 'label_encoder.pkl')
    joblib.dump(feature_names, 'feature_names.pkl')
    
    print("💾 Models saved successfully!")
    
    return rf_model, accuracy, feature_importance

def predict_new_sample(model, scaler, le, features, feature_names):
    """Predict a new sample"""
    # Create a DataFrame to ensure correct feature order
    features_df = pd.DataFrame([features], columns=feature_names)
    features_scaled = scaler.transform(features_df)
    
    prediction_encoded = model.predict(features_scaled)[0]
    confidence = np.max(model.predict_proba(features_scaled))
    prediction = le.inverse_transform([prediction_encoded])[0]
    
    return prediction, confidence

if __name__ == "__main__":
    csv_path = "star_classification.csv"  # Update with your file path
    
    try:
        model, accuracy, feature_importance = train_random_forest(csv_path)
        
        # Test with a sample prediction from the dataset
        print("\n🧪 Sample Prediction Test:")
        sample_features = [19.0, 18.5, 17.2, 16.8, 16.5, 0.1, 500, 55000, 123]
        # Features: [u, g, r, i, z, redshift, plate, MJD, fiberID]
        
        scaler = joblib.load('scaler.pkl')
        le = joblib.load('label_encoder.pkl')
        feature_names = joblib.load('feature_names.pkl')
        
        pred, conf = predict_new_sample(model, scaler, le, sample_features, feature_names)
        print(f"Sample Prediction: {pred} with confidence {conf:.4f}")
        
    except FileNotFoundError:
        print(f"❌ File not found: {csv_path}")
        print("Please make sure the CSV file is in the correct location.")