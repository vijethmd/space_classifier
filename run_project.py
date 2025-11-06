import subprocess
import sys
import os

def run_script(script_name, csv_path):
    """Run a Python script and capture its output"""
    print(f"\n{'='*60}")
    print(f"Running {script_name}")
    print(f"{'='*60}")
    
    try:
        # Pass the CSV path as an argument
        result = subprocess.run([sys.executable, script_name, csv_path], 
                              capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
        return True
    except Exception as e:
        print(f"Error running {script_name}: {e}")
        return False

def main():
    print("🚀 Starting SDSS Space Object Classifier Project!")
    
    # Get the CSV file path
    csv_path = "star_classification.csv"  # Update this if your file has a different name
    
    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found: {csv_path}")
        print("Please make sure the SDSS dataset file is in the project directory")
        print("You can download it from: https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17")
        return
    
    # Run all training scripts
    scripts = [
        'data_preparation.py',
        'random_forest_model.py', 
        'deep_learning_model.py',
        'cnn_model.py'
    ]
    
    for script in scripts:
        if os.path.exists(script):
            success = run_script(script, csv_path)
            if not success:
                print(f"Failed to run {script}, but continuing...")
        else:
            print(f"Script {script} not found!")
    
    print(f"\n{'='*60}")
    print("🎉 Training Complete!")
    print("To run the web app, use: streamlit run app.py")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()