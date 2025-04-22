import pandas as pd  # For reading and manipulating tabular data
from sklearn.ensemble import IsolationForest  # Isolation Forest model for anomaly detection
from sklearn.preprocessing import MinMaxScaler  # For feature normalization
import joblib  # For saving and loading trained ML models
import os  # For file and directory handling
from datetime import datetime  # For generating timestamps

# === CONFIG: Option1 Select preprocessed CSV ===
input_filename = "preprocessed_master_logs.csv"  # Name of the preprocessed input file
input_path = f"dataset/processed/{input_filename}"  # Full path to the preprocessed file

# === CONFIG: Output paths ===
output_dir = "dataset/model_output"  # Folder to save model and results
os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist

model_path = os.path.join(output_dir, "isolation_forest_model.pkl")  # Path to save the trained model
scored_output_path = os.path.join(output_dir, f"scored_{input_filename}")  # Path to save prediction results
log_path = os.path.join(output_dir, "training_log.txt")  # Log file to store training summary

# === Step 1: Load dataset ===
try:
    print(f"📂 Loading dataset: {input_path}")  # Notify user of loading step
    df = pd.read_csv(input_path)  # Load the preprocessed CSV into a DataFrame
except Exception as e:
    print(f"❌ Failed to load dataset: {e}")  # Print error if loading fails
    exit(1)  # Exit the script if data can't be loaded

# Keep a copy of the original DataFrame for later use
original_df = df.copy()

# === Step 2: Drop non-numeric fields ===
df_numeric = df.select_dtypes(include=["int64", "float64"])  # Select only numerical columns
if df_numeric.empty:  # If no numeric data is found
    print("❌ No numeric data to train on. Please check preprocessing.")
    exit()

# === Step 3: Normalize features ===
try:
    scaler = MinMaxScaler()  # Initialize MinMaxScaler to scale values between 0 and 1
    X_scaled = scaler.fit_transform(df_numeric)  # Apply scaling to numeric features
    # === Step 3.1: Save feature column names for real-time alignment ===
    feature_names = df_numeric.columns.tolist()
    joblib.dump(feature_names, os.path.join(output_dir, "isolation_feature_names.pkl"))

except Exception as e:
    print(f"❌ Failed to scale numeric features: {e}")
    exit(1)

# === Step 4: Train Isolation Forest ===
try:
    print("🧠 Training Isolation Forest model...")  # Notify training start
    model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)  # Initialize model
    model.fit(X_scaled)  # Train model on scaled features
except Exception as e:
    print(f"❌ Model training failed: {e}")
    exit(1)

# === Step 5: Generate predictions ===
try:
    scores = model.decision_function(X_scaled)  # Compute anomaly scores (higher = more normal)
    labels = model.predict(X_scaled)  # Predict anomalies (-1 for anomaly, 1 for normal)
except Exception as e:
    print(f"❌ Failed during model prediction: {e}")
    exit(1)

# === Step 6: Add anomaly results to original DataFrame ===
original_df["anomaly_score"] = scores  # Add anomaly score column
original_df["anomaly_label"] = (labels == -1).astype(int)  # Add label column (1 = anomaly, 0 = normal)

# === Step 7: Save trained model ===
try:
    joblib.dump(model, model_path)  # Serialize and save the trained model to disk
except Exception as e:
    print(f"❌ Failed to save model: {e}")
    exit(1)

# === Step 8: Save scored data ===
try:
    original_df.to_csv(scored_output_path, index=False)  # Save original data with scores and labels
except Exception as e:
    print(f"❌ Failed to save scored output: {e}")
    exit(1)

# === Step 9: Log summary ===
try:
    with open(log_path, "a") as log:  # Open log file in append mode
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Current timestamp
        log.write(f"[{now}] Trained on: {input_filename}\n")  # Log file name used for training
        log.write("Model: Isolation Forest (n_estimators=100, contamination=0.05)\n")  # Model config
        log.write(f"Anomalies Detected: {(labels == -1).sum()} / {len(labels)}\n")  # Number of anomalies
        log.write(f"Model saved to: {model_path}\n")  # Model path
        log.write(f"Scored output: {scored_output_path}\n")  # Output file path
        log.write("--------------------------------------------------------------------------------------------\n")
except Exception as e:
    print(f"⚠️ Failed to update training log: {e}")  # Warn if log writing fails

# === Step 10: Final Summary ===
print(f"✅ Model saved to: {model_path}")  # Confirm model was saved
print(f"📊 Scored data saved to: {scored_output_path}")  # Confirm output CSV was saved
print(f"🧠 Anomalies Detected: {(labels == -1).sum()} out of {len(labels)}")  # Print anomaly count
print(f"📝 Training log updated: {log_path}")  # Confirm log update
