import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
import time

print("🚀 Training Multiple Models (No TensorFlow - Fast & Reliable)...")
start_time = time.time()

# Load data
print("📊 Loading dataset...")
df = pd.read_csv("star_classification.csv")
print(f"✅ Data loaded: {df.shape}")

# Prepare features
features = ['u', 'g', 'r', 'i', 'z', 'redshift']
X = df[features]
y = df['class']

# Add engineered features
X_engineered = X.copy()
X_engineered['u_g'] = X['u'] - X['g']
X_engineered['g_r'] = X['g'] - X['r']
X_engineered['r_i'] = X['r'] - X['i']
X_engineered['i_z'] = X['i'] - X['z']

print(f"🎯 Features: {X_engineered.columns.tolist()}")

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"🏷️ Classes: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_engineered, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save preprocessing
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le, 'label_encoder.pkl')
joblib.dump(X_engineered.columns.tolist(), 'feature_names.pkl')
print("💾 Preprocessing objects saved!")

# ============================================================================
# TRAIN MULTIPLE SCICKIT-LEARN MODELS
# ============================================================================
models = {
    'rf': {
        'name': '🌲 Random Forest',
        'model': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=1),
        'accuracy': 0
    },
    'gb': {
        'name': '🎯 Gradient Boosting', 
        'model': GradientBoostingClassifier(n_estimators=100, random_state=42, verbose=1),
        'accuracy': 0
    },
    'svm': {
        'name': '⚡ Support Vector Machine',
        'model': SVC(probability=True, random_state=42, verbose=True),
        'accuracy': 0
    },
    'knn': {
        'name': '📊 K-Nearest Neighbors',
        'model': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        'accuracy': 0
    }
}

print("\n" + "="*60)
print("🤖 TRAINING MULTIPLE MACHINE LEARNING MODELS")
print("="*60)

results = []

for model_key, model_info in models.items():
    print(f"\n{model_info['name']}...")
    
    try:
        # Train model
        model_info['model'].fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model_info['model'].predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        model_info['accuracy'] = accuracy
        
        # Save model
        joblib.dump(model_info['model'], f'{model_key}_model.pkl')
        
        print(f"✅ {model_info['name']} Accuracy: {accuracy:.4f}")
        results.append((model_info['name'], accuracy))
        
    except Exception as e:
        print(f"❌ {model_info['name']} failed: {e}")

# ============================================================================
# RESULTS SUMMARY
# ============================================================================
print("\n" + "="*60)
print("📊 TRAINING COMPLETE - RESULTS SUMMARY")
print("="*60)

# Sort by accuracy
results.sort(key=lambda x: x[1], reverse=True)

print("\n🏆 Model Performance (Ranked):")
for i, (name, accuracy) in enumerate(results, 1):
    print(f"{i:2d}. {name}: {accuracy:.4f}")

# Test predictions
print("\n🧪 Sample Prediction Test:")
sample_features = [19.0, 18.5, 17.2, 16.8, 16.5, 0.1]
sample_df = pd.DataFrame([sample_features], columns=features)
sample_df['u_g'] = sample_df['u'] - sample_df['g']
sample_df['g_r'] = sample_df['g'] - sample_df['r'] 
sample_df['r_i'] = sample_df['r'] - sample_df['i']
sample_df['i_z'] = sample_df['i'] - sample_df['z']

sample_scaled = scaler.transform(sample_df)

print("Sample input:", sample_features)

for model_key, model_info in models.items():
    if model_info['accuracy'] > 0:  # Only if model trained successfully
        try:
            model = joblib.load(f'{model_key}_model.pkl')
            pred = model.predict(sample_scaled)[0]
            proba = model.predict_proba(sample_scaled)[0] if hasattr(model, 'predict_proba') else [1, 0, 0]
            confidence = np.max(proba)
            prediction = le.inverse_transform([pred])[0]
            
            print(f"  {model_info['name']}: {prediction} (confidence: {confidence:.4f})")
        except:
            pass

training_time = time.time() - start_time
print(f"\n⏱️ Total training time: {training_time:.2f} seconds")
print("🚀 Now run: streamlit run multi_model_comparison.py")