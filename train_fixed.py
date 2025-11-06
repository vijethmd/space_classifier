import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib

print("🚀 Training Fixed Random Forest Model...")

try:
    # Load the data
    df = pd.read_csv("star_classification.csv")
    print(f"✅ Data loaded: {df.shape}")
    
    # Show all columns to verify
    print("📋 Available columns:")
    for col in df.columns:
        print(f"  - {col}")
    
    # Common column names in SDSS dataset - try different variations
    possible_feature_sets = [
        # Try the most common column names
        ['u', 'g', 'r', 'i', 'z', 'redshift', 'plate', 'MJD', 'fiberID'],
        ['u', 'g', 'r', 'i', 'z', 'redshift', 'plate', 'MJD', 'fiberid'],
        ['u', 'g', 'r', 'i', 'z', 'redshift', 'plate', 'MJD', 'fiber_id'],
        # If fiberID doesn't exist, use just these
        ['u', 'g', 'r', 'i', 'z', 'redshift']
    ]
    
    # Find which feature set works
    selected_features = None
    for feature_set in possible_feature_sets:
        missing = [f for f in feature_set if f not in df.columns]
        if not missing:
            selected_features = feature_set
            print(f"🎯 Using features: {selected_features}")
            break
    
    if not selected_features:
        # If no standard set works, use what we have
        selected_features = [col for col in df.columns if col not in ['class', 'obj_ID', 'spec_obj_ID']]
        print(f"⚠️  Using available features: {selected_features}")
    
    X = df[selected_features]
    y = df['class']
    
    print(f"🎯 Classes: {y.unique()}")
    print(f"📊 Class distribution:\n{y.value_counts()}")

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print(f"🏷️ Label encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest
    print("\n🌲 Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = rf_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"🎯 Accuracy: {accuracy:.4f}")
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Save models
    joblib.dump(rf_model, 'random_forest_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(le, 'label_encoder.pkl')
    joblib.dump(selected_features, 'feature_names.pkl')

    print("💾 Models saved successfully!")
    print("\n🎉 Now run: streamlit run simple_app.py")

except Exception as e:
    print(f"❌ Error: {e}")
    print("\nLet's try a super simple version with just basic features...")
    
    # Super simple fallback
    try:
        df = pd.read_csv("star_classification.csv")
        # Use only the most basic features that should definitely exist
        basic_features = ['u', 'g', 'r', 'i', 'z']
        basic_features = [f for f in basic_features if f in df.columns]
        
        if not basic_features:
            basic_features = [col for col in df.columns if col not in ['class']][:5]
            
        print(f"🔄 Trying with basic features: {basic_features}")
        
        X = df[basic_features]
        y = df['class']
        
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        accuracy = accuracy_score(y_test, model.predict(X_test_scaled))
        print(f"🎯 Basic model accuracy: {accuracy:.4f}")
        
        joblib.dump(model, 'random_forest_model.pkl')
        joblib.dump(scaler, 'scaler.pkl')
        joblib.dump(le, 'label_encoder.pkl')
        joblib.dump(basic_features, 'feature_names.pkl')
        
        print("💾 Basic models saved!")
        
    except Exception as e2:
        print(f"❌ Even basic version failed: {e2}")