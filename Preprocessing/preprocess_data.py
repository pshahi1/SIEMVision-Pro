import pandas as pd  # For data manipulation and DataFrame operations
import os  # For working with file paths and directories
from sklearn.preprocessing import LabelEncoder, MinMaxScaler  # For encoding and normalizing features
from datetime import datetime  # For generating timestamps for logging

# ==== Step 1: Manual Selection of Input File ====

input_filename = "master_logs.csv"  # Replace with your desired raw CSV file
input_path = "dataset/raw/" + input_filename  # Full path to the input file
output_dir = "dataset/processed"  # Directory where the preprocessed file will be saved
output_filename = f"preprocessed_{input_filename}"  # Output filename with a prefix
output_path = f"{output_dir}/{output_filename}"  # Full path to save the output CSV

# Create the processed dataset directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# ==== Step 2: Load CSV with Error Handling ====
try:
    print(f"📂 Loading: {input_path}")  # Show the input file being loaded
    df = pd.read_csv(input_path)  # Read the CSV into a DataFrame
except Exception as e:
    print(f"❌ Failed to load CSV file: {e}")  # Show error if loading fails
    exit(1)  # Stop script if file loading fails

# ==== Step 3: Drop rows with too many missing fields ====
df.dropna(thresh=4, inplace=True)  # Drop rows that have fewer than 4 non-null values

# ==== Step 4: Handle Timestamp - Extract hour:minute:second ====
if "@timestamp" in df.columns:  # Check if timestamp column exists
    try:
        df["@timestamp"] = pd.to_datetime(df["@timestamp"], errors="coerce")  # Convert to datetime format
        df["hour_minute_second"] = df["@timestamp"].dt.strftime("%H:%M:%S")  # Extract time portion
        df.drop(columns=["@timestamp"], inplace=True)  # Drop original timestamp column
    except Exception as e:
        print(f"⚠️ Failed to process timestamp: {e}")  # Warn if parsing fails

# ==== Step 5: Label Encode All Categorical Columns ====
label_encoders = {}  # Dictionary to store encoders for later use or reverse transformation
print("🔍 Starting label encoding of categorical fields...")

for col in df.columns:
    if df[col].dtype == "object":  # Check if column is categorical
        try:
            df[col] = df[col].astype(str)  # Ensure values are strings
            le = LabelEncoder()  # Create label encoder
            df[col] = le.fit_transform(df[col])  # Transform categorical values to numerical
            label_encoders[col] = le  # Save encoder
            print(f"🔢 Encoded: {col}")
        except Exception as e:
            print(f"⚠️ Could not encode '{col}': {e}")  # Warn if encoding fails
    else:
        print(f"✅ Skipped non-categorical field: {col}")  # Skip numeric fields

# ==== Step 6: Normalize Numerical Columns ====
scaler = MinMaxScaler()  # Initialize Min-Max scaler
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns  # Select numeric columns

print("📐 Normalizing numerical columns...")
for col in numeric_cols:
    try:
        df[[col]] = scaler.fit_transform(df[[col]])  # Normalize column to 0–1 range
        print(f"🔄 Normalized: {col}")
    except Exception as e:
        print(f"⚠️ Could not normalize '{col}': {e}")  # Warn if normalization fails

# ==== Step 7: Fill Any Remaining NaNs ====
df.fillna("unknown", inplace=True)  # Fill any leftover missing values with the string "unknown"

# ==== Step 8: Save Preprocessed Dataset ====
try:
    df.to_csv(output_path, index=False)  # Save the cleaned DataFrame to CSV
    print(f"✅ Preprocessed file saved to: {output_path}")
except Exception as e:
    print(f"❌ Failed to save CSV: {e}")  # Error if file saving fails

# ==== Step 9: Save Preprocessing Metadata ====
log_path = f"{output_dir}/preprocess_log.txt"  # Define path to preprocessing log
log_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Current time for log

try:
    with open(log_path, "a") as log_file:  # Open log file in append mode
        log_file.write(f"{output_filename} | from: {input_filename} | {log_time}\n")  # Record metadata
except Exception as e:
    print(f"⚠️ Failed to write log file: {e}")  # Error handling for logging
