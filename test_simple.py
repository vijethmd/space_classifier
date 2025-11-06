import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

print("🧪 Testing basic functionality...")

# Try to load the data
try:
    df = pd.read_csv("star_classification.csv")
    print(f"✅ Data loaded successfully! Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Class distribution:\n{df['class'].value_counts()}")
    
    # Simple model test
    X = df[['u', 'g', 'r', 'i', 'z', 'redshift']]
    y = df['class']
    
    # Encode labels
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    # Train a simple model
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"✅ Simple model accuracy: {accuracy:.4f}")
    
    # Test plotting
    plt.figure(figsize=(8, 6))
    plt.scatter(X['g'] - X['r'], X['u'] - X['g'], c=y_encoded, alpha=0.5)
    plt.xlabel('g - r')
    plt.ylabel('u - g')
    plt.title('Simple Color-Color Plot')
    plt.savefig('test_plot.png')
    print("✅ Plot created successfully!")
    
except Exception as e:
    print(f"❌ Error: {e}")