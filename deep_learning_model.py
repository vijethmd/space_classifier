import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from data_preparation import prepare_data

def create_deep_model(input_dim, num_classes):
    """Create a deep neural network for SDSS data"""
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

def train_deep_learning_model(csv_file_path):
    """Train and evaluate a deep learning model on SDSS data"""
    print("=== Training Deep Learning Model ===")
    
    # Prepare data
    X_train, X_test, y_train, y_test, le, scaler, feature_names = prepare_data(csv_file_path)
    
    # Create model
    model = create_deep_model(X_train.shape[1], len(le.classes_))
    
    print("🧠 Model Architecture:")
    model.summary()
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10)
    ]
    
    # Train the model
    print("Training Deep Learning Model...")
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=64,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"🎯 Deep Learning Model Accuracy: {test_accuracy:.4f}")
    
    # Make predictions
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save the model
    model.save('deep_learning_model.h5')
    print("💾 Deep Learning model saved!")
    
    return model, test_accuracy, history

if __name__ == "__main__":
    csv_path = "star_classification.csv"  # Update with your file path
    
    try:
        model, accuracy, history = train_deep_learning_model(csv_path)
    except FileNotFoundError:
        print(f"❌ File not found: {csv_path}")