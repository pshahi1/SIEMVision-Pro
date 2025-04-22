# Importing required libraries
import pandas as pd  # For reading and manipulating data
import numpy as np  # For numerical operations, especially for error calculations
import os  # For file system operations
from datetime import datetime  # For logging with timestamp
from sklearn.preprocessing import MinMaxScaler  # To scale input data to 0-1 range
from tensorflow.keras.models import Model  # Base Keras model class
from tensorflow.keras.layers import Input, Dense  # Layers for building the autoencoder
from tensorflow.keras.callbacks import EarlyStopping  # Stop training if model stops improving
import tensorflow as tf  # TensorFlow backend

# === CONFIG: Option 1: Input/Output paths ===
input_filename = "preprocessed_master_logs.csv"  # Preprocessed input file name
input_path = f"dataset/processed/{input_filename}"  # Full input file path
output_dir = "dataset/model_output"  # Directory to store model and output files
os.makedirs(output_dir, exist_ok=True)  # Create output directory if not exists

# Define output file paths
model_path = os.path.join(output_dir, "autoencoder_model.h5")  # File path to save trained model
scored_output_path = os.path.join(output_dir, f"scored_autoencoder_{input_filename}")  # Path to save output with scores
log_path = os.path.join(output_dir, "training_log.txt")  # Path to store training logs

# === Step 1: Load Dataset ===
try:
    print(f"📂 Loading dataset: {input_path}")  # Notify that dataset loading has started
    df = pd.read_csv(input_path)  # Load the preprocessed CSV into a DataFrame
except Exception as e:
    print(f"❌ Failed to load dataset: {e}")  # Catch any error during loading
    exit(1)

original_df = df.copy()  # Keep a copy of original data to append scores later

# === Step 2: Select Numeric Features ===
df_numeric = df.select_dtypes(include=["int64", "float64"])  # Keep only numeric columns
if df_numeric.empty:
    print("❌ No numeric features found. Please check preprocessing.")  # Exit if no numeric features
    exit()

# === Step 3: Normalize Features ===
try:
    scaler = MinMaxScaler()  # Initialize MinMaxScaler
    X_scaled = scaler.fit_transform(df_numeric)  # Normalize numeric data to 0–1
except Exception as e:
    print(f"❌ Failed to normalize features: {e}")  # Catch normalization errors
    exit()

# === Step 4: Define Autoencoder Architecture ===
input_dim = X_scaled.shape[1]  # Get number of input features
encoding_dim = input_dim // 2  # Set size of encoded layer (compressed)

# Build the autoencoder model
input_layer = Input(shape=(input_dim,))  # Input layer
encoded = Dense(encoding_dim, activation="relu")(input_layer)  # Encoder layer
decoded = Dense(input_dim, activation="sigmoid")(encoded)  # Decoder layer

autoencoder = Model(inputs=input_layer, outputs=decoded)  # Build the model
autoencoder.compile(optimizer="adam", loss="mse")  # Compile the model with MSE loss

# === Step 5: Train Autoencoder ===
try:
    print("🧠 Training Autoencoder model...")  # Notify start of training
    early_stop = EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)  # Stop if loss doesn't improve
    autoencoder.fit(
        X_scaled, X_scaled,  # Input = Output for unsupervised learning
        epochs=50,  # Max number of training epochs
        batch_size=32,  # Number of samples per gradient update
        shuffle=True,  # Shuffle data before each epoch
        callbacks=[early_stop],  # Add early stopping
        verbose=1  # Show training progress
    )
except Exception as e:
    print(f"❌ Model training failed: {e}")  # Catch training failure
    exit(1)

# === Step 6: Reconstruction Error Calculation ===
try:
    print("📊 Calculating reconstruction error...")  # Notify scoring step
    reconstructions = autoencoder.predict(X_scaled)  # Get reconstructed outputs
    mse = np.mean(np.power(X_scaled - reconstructions, 2), axis=1)  # Mean Squared Error for each input
    threshold = np.percentile(mse, 95)  # Set threshold at 95th percentile for anomaly detection
except Exception as e:
    print(f"❌ Error during scoring: {e}")  # Handle prediction error
    exit(1)

# === Step 7: Append Scores and Labels to Original Data ===
original_df["anomaly_score"] = mse  # Add reconstruction error as score
original_df["anomaly_label"] = (mse > threshold).astype(int)  # Label anomalies (1 = anomaly, 0 = normal)

# === Step 8: Save Model and Output ===
try:
    autoencoder.save(model_path)  # Save trained autoencoder model
    original_df.to_csv(scored_output_path, index=False)  # Save scored data
except Exception as e:
    print(f"❌ Failed to save model or output: {e}")  # Handle save errors
    exit(1)

# === Step 9: Update Training Log ===
try:
    with open(log_path, "a") as log:  # Open log file in append mode
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Get current timestamp
        log.write(f"[{now}] Trained on: {input_filename}\n")  # Log input file
        log.write("Model: Autoencoder (Keras, Dense, relu-sigmoid, EarlyStopping)\n")  # Model details
        log.write(f"Anomalies Detected: {original_df['anomaly_label'].sum()} / {len(original_df)}\n")  # Anomaly count
        log.write(f"Model saved to: {model_path}\n")  # Log model path
        log.write(f"Scored output: {scored_output_path}\n")  # Log output path
        log.write("---------------------------------------------------------------------------------------------\n")
except Exception as e:
    print(f"⚠️ Failed to write training log: {e}")  # Handle log errors

# === Final Status ===
print(f"✅ Autoencoder model saved to: {model_path}")  # Confirm model saved
print(f"📊 Scored data saved to: {scored_output_path}")  # Confirm output saved
print(f"🧠 Anomalies Detected: {original_df['anomaly_label'].sum()} out of {len(original_df)}")  # Show anomaly stats
print(f"📝 Training log updated: {log_path}")  # Confirm log update
