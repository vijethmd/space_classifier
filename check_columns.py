import pandas as pd

# Load the data and check columns
df = pd.read_csv("star_classification.csv")
print("📊 All columns in your CSV file:")
for i, col in enumerate(df.columns):
    print(f"  {i+1:2d}. '{col}'")

print(f"\n✅ First few rows of the 'class' column:")
print(df['class'].value_counts())

print(f"\n🔍 Sample of the data:")
print(df.head(3))