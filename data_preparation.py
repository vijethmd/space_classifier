import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt

def load_and_explore_data(csv_file_path):
    """
    Load and explore the SDSS stellar classification dataset
    """
    print("📊 Loading dataset...")
    df = pd.read_csv(csv_file_path)
    
    print(f"Dataset shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head())
    
    print("\nColumn names:")
    print(df.columns.tolist())
    
    print("\nClass distribution:")
    print(df['class'].value_counts())
    
    return df

def preprocess_data(df):
    """
    Preprocess the SDSS dataset for machine learning
    """
    print("\n🔧 Preprocessing data...")
    
    # Check for missing values
    print("Missing values:")
    print(df.isnull().sum())
    
    # Select relevant features for classification
    feature_columns = [
        'u', 'g', 'r', 'i', 'z',  # Photometric magnitudes
        'redshift',                # Redshift
        'plate',                   # Plate number
        'MJD',                     # Modified Julian Date
        'fiberID'                  # Fiber ID
    ]
    
    # Create feature matrix and target
    X = df[feature_columns]
    y = df['class']  # This should be 'GALAXY', 'STAR', 'QSO'
    
    print(f"Features: {feature_columns}")
    print(f"Target classes: {y.unique()}")
    print(f"Class distribution:\n{y.value_counts()}")
    
    return X, y

def explore_features(X, y):
    """
    Create some visualizations to understand the data
    """
    print("\n📈 Creating visualizations...")
    
    # Create a DataFrame for plotting
    plot_df = X.copy()
    plot_df['class'] = y
    
    # Plot feature distributions by class
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Color-color plot (g-r vs u-g) - common in astronomy
    plt.subplot(2, 2, 1)
    for class_type in plot_df['class'].unique():
        class_data = plot_df[plot_df['class'] == class_type]
        plt.scatter(class_data['g'] - class_data['r'], 
                   class_data['u'] - class_data['g'], 
                   alpha=0.6, label=class_type, s=10)
    plt.xlabel('g - r (Color)')
    plt.ylabel('u - g (Color)')
    plt.legend()
    plt.title('Color-Color Diagram')
    
    # Plot 2: Redshift distribution
    plt.subplot(2, 2, 2)
    for class_type in plot_df['class'].unique():
        class_data = plot_df[plot_df['class'] == class_type]
        plt.hist(class_data['redshift'], alpha=0.7, label=class_type, bins=50)
    plt.xlabel('Redshift')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('Redshift Distribution')
    
    # Plot 3: r-band magnitude distribution
    plt.subplot(2, 2, 3)
    for class_type in plot_df['class'].unique():
        class_data = plot_df[plot_df['class'] == class_type]
        plt.hist(class_data['r'], alpha=0.7, label=class_type, bins=50)
    plt.xlabel('r-band magnitude')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('r-band Magnitude Distribution')
    
    # Plot 4: Class distribution
    plt.subplot(2, 2, 4)
    y.value_counts().plot(kind='bar')
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('data_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Visualizations saved as 'data_visualization.png'")

def prepare_data(csv_file_path, test_size=0.2, random_state=42):
    """
    Main function to prepare data for training
    """
    # Load data
    df = load_and_explore_data(csv_file_path)
    
    # Preprocess data
    X, y = preprocess_data(df)
    
    # Create visualizations
    explore_features(X, y)
    
    # Encode labels (GALAXY -> 0, QSO -> 1, STAR -> 2)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    print(f"Label encoding: {dict(zip(le.classes_, range(len(le.classes_))))}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\n📊 Data preparation complete:")
    print(f"Training set: {X_train_scaled.shape}")
    print(f"Test set: {X_test_scaled.shape}")
    print(f"Features: {X.columns.tolist()}")
    print(f"Classes: {le.classes_}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, le, scaler, X.columns

if __name__ == "__main__":
    # Update this path to where your CSV file is located
    csv_path = "star_classification.csv"  # or whatever your file is named
    
    try:
        X_train, X_test, y_train, y_test, le, scaler, feature_names = prepare_data(csv_path)
        print("✅ Data preparation successful!")
    except FileNotFoundError:
        print(f"❌ File not found: {csv_path}")
        print("Please make sure the CSV file is in the same directory or update the path.")