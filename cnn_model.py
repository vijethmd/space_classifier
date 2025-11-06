import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from data_preparation import prepare_data

def create_1d_cnn_model(input_dim, num_classes):
    """Create a 1D CNN model for tabular data"""
    model = keras.Sequential([
        # Reshape for 1D convolution (batch_size, timesteps, features)
        layers.Reshape((input_dim, 1), input_shape=(input_dim,)),
        
        # First 1D convolutional block
        layers.Conv1D(64, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.3),
        
        # Second 1D convolutional block
        layers.Conv1D(128, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.3),
        
        # Third 1D convolutional block
        layers.Conv1D(256, 2, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.4),
        
        # Classifier
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

def train_cnn_model(csv_file_path):
    """Train and evaluate the 1D CNN model"""
    print("=== Training 1D CNN Model ===")
    
    # Prepare data
    X_train, X_test, y_train, y_test, le, scaler, feature_names = prepare_data(csv_file_path)
    
    # Create model
    model = create_1d_cnn_model(X_train.shape[1], len(le.classes_))
    
    print("📷 1D CNN Model Architecture:")
    model.summary()
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10)
    ]
    
    # Train the model
    print("Training 1D CNN Model...")
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=128,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"🎯 1D CNN Model Accuracy: {test_accuracy:.4f}")
    
    # Make predictions
    y_pred = np.argmax(model.predict(X_test), axis=1)
    
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('1D CNN Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('1D CNN Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('cnn_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save the model
    model.save('cnn_model.h5')
    print("💾 CNN model saved!")
    
    return model, test_accuracy, history

if __name__ == "__main__":
    csv_path = "star_classification.csv"  # Update with your file path
    
    try:
        model, accuracy, history = train_cnn_model(csv_path)
    except FileNotFoundError:
        print(f"❌ File not found: {csv_path}")