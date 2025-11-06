import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
import time

print("🚀 Starting Reliable Training (One Model at a Time)...")

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
# 1. RANDOM FOREST (ALWAYS WORKS)
# ============================================================================
print("\n" + "="*60)
print("🌲 TRAINING RANDOM FOREST MODEL")
print("="*60)

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

print("Training Random Forest...")
rf_model.fit(X_train_scaled, y_train)

rf_pred = rf_model.predict(X_test_scaled)
rf_accuracy = accuracy_score(y_test, rf_pred)

print(f"🎯 Random Forest Accuracy: {rf_accuracy:.4f}")
joblib.dump(rf_model, 'random_forest_model.pkl')
print("💾 Random Forest model saved!")

# ============================================================================
# 2. DEEP LEARNING MODELS (OPTIONAL - SKIP IF THEY FAIL)
# ============================================================================
print("\n" + "="*60)
print("🤖 ATTEMPTING DEEP LEARNING MODELS")
print("="*60)

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    
    print("✅ TensorFlow imported successfully!")
    
    # ============================================================================
    # 2A. DEEP NEURAL NETWORK
    # ============================================================================
    print("\n🧠 Training Deep Neural Network...")
    
    def create_simple_dnn(input_dim, num_classes):
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(input_dim,)),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    dnn_model = create_simple_dnn(X_train_scaled.shape[1], len(le.classes_))
    
    # Quick training with fewer epochs
    dnn_history = dnn_model.fit(
        X_train_scaled, y_train,
        epochs=20,
        batch_size=128,
        validation_split=0.2,
        verbose=1
    )
    
    dnn_loss, dnn_accuracy = dnn_model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"🎯 DNN Accuracy: {dnn_accuracy:.4f}")
    
    dnn_model.save('dnn_model.h5')
    print("💾 DNN model saved!")
    
    # ============================================================================
    # 2B. 1D CNN
    # ============================================================================
    print("\n📷 Training 1D CNN...")
    
    # Reshape for CNN
    X_train_cnn = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
    X_test_cnn = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)
    
    def create_simple_cnn(input_shape, num_classes):
        model = keras.Sequential([
            layers.Conv1D(32, 3, activation='relu', input_shape=input_shape),
            layers.MaxPooling1D(2),
            layers.Dropout(0.3),
            
            layers.Conv1D(64, 3, activation='relu'),
            layers.GlobalAveragePooling1D(),
            layers.Dropout(0.3),
            
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    cnn_model = create_simple_cnn((X_train_scaled.shape[1], 1), len(le.classes_))
    
    cnn_history = cnn_model.fit(
        X_train_cnn, y_train,
        epochs=20,
        batch_size=128,
        validation_split=0.2,
        verbose=1
    )
    
    cnn_loss, cnn_accuracy = cnn_model.evaluate(X_test_cnn, y_test, verbose=0)
    print(f"🎯 CNN Accuracy: {cnn_accuracy:.4f}")
    
    cnn_model.save('cnn_model.h5')
    print("💾 CNN model saved!")
    
    print("\n🎉 ALL MODELS TRAINED SUCCESSFULLY!")
    print("📊 Final Accuracies:")
    print(f"   🌲 Random Forest: {rf_accuracy:.4f}")
    print(f"   🧠 DNN: {dnn_accuracy:.4f}")
    print(f"   📷 CNN: {cnn_accuracy:.4f}")

except Exception as e:
    print(f"⚠️ Deep Learning models failed: {e}")
    print("But don't worry! Random Forest is working and that's the most important one!")
    print("You can still use the web app with Random Forest.")

print("\n" + "="*60)
print("🚀 TRAINING COMPLETE!")
print("="*60)
print("Now run: streamlit run multi_model_app.py")
print("The web app will use whatever models are available.")