import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import matplotlib.pyplot as plt

print("🚀 Training All Three Models...")

# Load data
df = pd.read_csv("star_classification.csv")

print("📊 Dataset Info:")
print(f"Shape: {df.shape}")
print(f"Classes: {df['class'].value_counts()}")

# Prepare features with engineering
features = ['u', 'g', 'r', 'i', 'z', 'redshift']
X = df[features]
y = df['class']

# Add engineered features
X_engineered = X.copy()
X_engineered['u_g'] = X['u'] - X['g']
X_engineered['g_r'] = X['g'] - X['r']
X_engineered['r_i'] = X['r'] - X['i']
X_engineered['i_z'] = X['i'] - X['z']

print(f"🎯 Using features: {X_engineered.columns.tolist()}")

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_engineered, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save preprocessing objects
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le, 'label_encoder.pkl')
joblib.dump(X_engineered.columns.tolist(), 'feature_names.pkl')

print("💾 Preprocessing objects saved!")

# ============================================================================
# 1. RANDOM FOREST MODEL
# ============================================================================
print("\n" + "="*50)
print("🌲 Training Random Forest Model...")
print("="*50)

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)
rf_accuracy = accuracy_score(y_test, rf_pred)

print(f"🎯 Random Forest Accuracy: {rf_accuracy:.4f}")
joblib.dump(rf_model, 'random_forest_model.pkl')
print("💾 Random Forest model saved!")

# ============================================================================
# 2. DEEP NEURAL NETWORK MODEL
# ============================================================================
print("\n" + "="*50)
print("🧠 Training Deep Neural Network Model...")
print("="*50)

def create_dnn_model(input_dim, num_classes):
    model = keras.Sequential([
        layers.Dense(256, activation='relu', input_shape=(input_dim,)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        
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

dnn_model = create_dnn_model(X_train_scaled.shape[1], len(le.classes_))

# Callbacks
callbacks = [
    keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10)
]

print("Training Deep Neural Network...")
dnn_history = dnn_model.fit(
    X_train_scaled, y_train,
    epochs=50,
    batch_size=64,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

dnn_loss, dnn_accuracy = dnn_model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"🎯 Deep Neural Network Accuracy: {dnn_accuracy:.4f}")

dnn_model.save('dnn_model.h5')
print("💾 Deep Neural Network model saved!")

# ============================================================================
# 3. 1D CNN MODEL
# ============================================================================
print("\n" + "="*50)
print("📷 Training 1D CNN Model...")
print("="*50)

def create_1d_cnn_model(input_dim, num_classes):
    model = keras.Sequential([
        layers.Reshape((input_dim, 1), input_shape=(input_dim,)),
        
        layers.Conv1D(64, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.3),
        
        layers.Conv1D(128, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.3),
        
        layers.Conv1D(256, 2, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.4),
        
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Reshape data for CNN
X_train_cnn = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
X_test_cnn = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

cnn_model = create_1d_cnn_model(X_train_scaled.shape[1], len(le.classes_))

print("Training 1D CNN...")
cnn_history = cnn_model.fit(
    X_train_cnn, y_train,
    epochs=50,
    batch_size=128,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

cnn_loss, cnn_accuracy = cnn_model.evaluate(X_test_cnn, y_test, verbose=0)
print(f"🎯 1D CNN Accuracy: {cnn_accuracy:.4f}")

cnn_model.save('cnn_model.h5')
print("💾 1D CNN model saved!")

# ============================================================================
# FINAL RESULTS COMPARISON
# ============================================================================
print("\n" + "="*50)
print("📊 FINAL MODEL COMPARISON")
print("="*50)

print(f"🌲 Random Forest Accuracy:     {rf_accuracy:.4f}")
print(f"🧠 Deep Neural Network Accuracy: {dnn_accuracy:.4f}")
print(f"📷 1D CNN Accuracy:            {cnn_accuracy:.4f}")

# Test with sample predictions
print("\n🧪 Sample Prediction Test:")
sample_features = [19.0, 18.5, 17.2, 16.8, 16.5, 0.1]  # Basic features
sample_df = pd.DataFrame([sample_features], columns=features)
sample_df['u_g'] = sample_df['u'] - sample_df['g']
sample_df['g_r'] = sample_df['g'] - sample_df['r']
sample_df['r_i'] = sample_df['r'] - sample_df['i']
sample_df['i_z'] = sample_df['i'] - sample_df['z']

sample_scaled = scaler.transform(sample_df)
sample_cnn = sample_scaled.reshape(sample_scaled.shape[0], sample_scaled.shape[1], 1)

# RF prediction
rf_pred = rf_model.predict(sample_scaled)[0]
rf_conf = np.max(rf_model.predict_proba(sample_scaled))

# DNN prediction  
dnn_pred_proba = dnn_model.predict(sample_scaled)
dnn_pred = np.argmax(dnn_pred_proba, axis=1)[0]
dnn_conf = np.max(dnn_pred_proba)

# CNN prediction
cnn_pred_proba = cnn_model.predict(sample_cnn)
cnn_pred = np.argmax(cnn_pred_proba, axis=1)[0]
cnn_conf = np.max(cnn_pred_proba)

print(f"Sample Input: {sample_features}")
print(f"🌲 RF Prediction:   {le.inverse_transform([rf_pred])[0]} (confidence: {rf_conf:.4f})")
print(f"🧠 DNN Prediction:  {le.inverse_transform([dnn_pred])[0]} (confidence: {dnn_conf:.4f})")
print(f"📷 CNN Prediction:  {le.inverse_transform([cnn_pred])[0]} (confidence: {cnn_conf:.4f})")

print("\n🎉 All models trained and saved successfully!")
print("🚀 Now run: streamlit run multi_model_app.py")