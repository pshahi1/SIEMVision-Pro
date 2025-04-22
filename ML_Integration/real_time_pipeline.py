# === Import Required Libraries ===
import pandas as pd  # Used for DataFrame creation and manipulation
import numpy as np  # For numerical operations and array handling
import joblib  # For loading the saved Isolation Forest model and feature names
import os  # To work with file and directory paths
import time  # To implement delays between log fetches in the loop
import warnings  # To suppress warnings that might clutter the output
from datetime import datetime  # For timestamping logs and outputs
from elasticsearch import Elasticsearch, helpers  # To connect to Elasticsearch and perform bulk operations
from sklearn.preprocessing import MinMaxScaler  # For normalizing numeric feature values
from tensorflow.keras.models import load_model  # To load the trained Keras Autoencoder model

# Suppress all warnings (e.g., FutureWarnings, UserWarnings)
warnings.filterwarnings("ignore")

# === Configuration Constants ===
ES_HOST = "https://192.168.1.69:9200"  # URL of the Elasticsearch instance
ES_USER = "elastic"  # Elasticsearch username
ES_PASSWORD = "HlFTDthZxj1uSqBvDbEJ"  # Elasticsearch password
ES_INDEX_INPUT = "winlogbeat-*"  # Index pattern to read logs from
ES_INDEX_OUTPUT = "anomalies-siemvision"  # Index to write anomaly detection results
FETCH_INTERVAL = 10  # Time (in seconds) between consecutive log fetches
FETCH_SIZE = 1000  # Number of logs to fetch in each cycle

# === Load Trained ML Models and Feature Names ===
print("📦 Loading trained models...")
try:
    isolation_model = joblib.load("dataset/model_output/isolation_forest_model.pkl")  # Load Isolation Forest model
    autoencoder_model = load_model("dataset/model_output/autoencoder_model.h5", compile=False)  # Load Autoencoder model
    feature_names = joblib.load("dataset/model_output/isolation_feature_names.pkl")  # Load feature names used during training
except Exception as e:
    print(f"❌ Failed to load models or feature names: {e}")
    exit(1)  # Exit if models or features can't be loaded

# === Connect to Elasticsearch ===
print("🔌 Connecting to Elasticsearch...")
try:
    es = Elasticsearch(
        ES_HOST,  # Host address
        basic_auth=(ES_USER, ES_PASSWORD),  # Authentication credentials
        verify_certs=False  # Ignore SSL certificate verification (for local/test use)
    )
except Exception as e:
    print(f"❌ Failed to connect to Elasticsearch: {e}")
    exit(1)  # Exit if connection fails

# === Function: Fetch logs from Elasticsearch ===
def fetch_logs():
    try:
        query = {
            "size": FETCH_SIZE,  # Limit number of logs
            "_source": [  # Specify fields to retrieve
                "event.code", "event.action", "event.original", "agent.name", "@timestamp"
            ],
            "query": {
                "bool": {
                    "must": [
                        {"match": {"event.provider": "Microsoft-Windows-Sysmon"}}  # Only fetch Sysmon logs
                    ]
                }
            }
        }
        res = es.search(index=ES_INDEX_INPUT, query=query["query"], _source=query["_source"], size=query["size"])  # Run search
        hits = [hit["_source"] for hit in res["hits"]["hits"]]  # Extract _source field from each hit
        return pd.DataFrame(hits)  # Return logs as DataFrame
    except Exception as e:
        print(f"❌ Failed to fetch logs: {e}")
        return pd.DataFrame()  # Return empty DataFrame on failure

# === Fetch and Preview Logs ===
print("🔍 Fetching logs...")
logs_df = fetch_logs()  # Fetch logs initially for preview

if logs_df.empty:
    print("⚠️ No logs returned.")  # Notify if no logs were fetched
else:
    print(f"📄 Retrieved {len(logs_df)} logs.")  # Show number of logs fetched

# === Function: Preprocess logs for ML model input ===
def preprocess(df):
    try:
        if df.empty:
            return None, None  # If input is empty, return None

        df_copy = df.copy()  # Work on a copy to avoid modifying original

        # Extract relevant fields from nested dictionaries
        df_copy["agent.name"] = df_copy["agent"].apply(lambda x: x.get("name") if isinstance(x, dict) else str(x))
        df_copy["event.code"] = df_copy["event"].apply(lambda x: x.get("code") if isinstance(x, dict) else str(x))
        df_copy["event.action"] = df_copy["event"].apply(lambda x: x.get("action") if isinstance(x, dict) else str(x))
        df_copy["event.original"] = df_copy["event"].apply(lambda x: x.get("original") if isinstance(x, dict) else str(x))

        df_copy.drop(columns=["agent", "event"], inplace=True, errors="ignore")  # Drop original nested fields

        # Convert timestamp to datetime and extract seconds since midnight
        df_copy["@timestamp"] = pd.to_datetime(df_copy["@timestamp"], errors="coerce")
        df_copy["hour_minute_second"] = (
            df_copy["@timestamp"].dt.hour * 3600 +
            df_copy["@timestamp"].dt.minute * 60 +
            df_copy["@timestamp"].dt.second
        )
        df_copy.drop(columns=["@timestamp"], inplace=True)  # Remove original timestamp field

        # Ensure categorical fields are strings and apply one-hot encoding
        fields = ["event.code", "event.action", "event.original", "agent.name"]
        for field in fields:
            df_copy[field] = df_copy[field].apply(lambda x: str(x) if x is not None else "missing")

        df_encoded = pd.get_dummies(df_copy, columns=fields)  # Apply one-hot encoding

        # Add any missing columns from training and re-order
        for col in feature_names:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        df_encoded = df_encoded[feature_names]

        # Scale features to [0, 1] range using MinMaxScaler
        df_scaled = MinMaxScaler().fit_transform(df_encoded)
        return df, df_scaled  # Return original+scaled features
    except Exception as e:
        print(f"❌ Error during preprocessing: {e}")
        return None, None

# === Preprocess logs (test run) ===
print("🔍 Fetching logs for preprocessing...")
logs_df = fetch_logs()  # Re-fetch logs for preprocessing
raw_df, X = preprocess(logs_df)  # Run preprocessing

if raw_df is not None and X is not None:
    print(f"✅ Preprocessing complete. Feature shape: {X.shape}")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    pd.set_option("display.precision", 6)
    print(pd.DataFrame(X).head(5))  # Show sample of preprocessed features
else:
    print("⚠️ Preprocessing failed.")

# === Severity Level Assignment Logic ===
def assign_severity(score, method="if"):
    if method == "if":  # For Isolation Forest
        if score > 0.7:
            return "Low"
        elif score > 0.3:
            return "Medium"
        else:
            return "High"
    elif method == "ae":  # For Autoencoder
        if score < 0.3:
            return "Low"
        elif score < 0.7:
            return "Medium"
        else:
            return "High"
    return "Unknown"  # Fallback

# === Predict Anomalies and Push to Elasticsearch ===
def predict_and_output(raw_df, X):
    try:
        if raw_df is None or X is None:
            print("⚠️ No data to score.")
            return

        # Predict using Isolation Forest
        iso_scores = isolation_model.decision_function(X)
        iso_labels = isolation_model.predict(X)
        raw_df["iso_score"] = iso_scores
        raw_df["iso_anomaly"] = (iso_labels == -1).astype(int)
        raw_df["iso_severity"] = [assign_severity(s, method="if") for s in iso_scores]

        # Predict using Autoencoder
        reconstructions = autoencoder_model.predict(X)
        mse = np.mean(np.power(X - reconstructions, 2), axis=1)
        threshold = np.percentile(mse, 95)  # Set anomaly threshold
        ae_labels = (mse > threshold).astype(int)

        raw_df["ae_score"] = mse
        raw_df["ae_anomaly"] = ae_labels
        raw_df["ae_severity"] = [assign_severity(s, method="ae") for s in mse]

        # Add current timestamp for each record
        timestamp = datetime.utcnow().isoformat()
        raw_df["@timestamp"] = timestamp

        # Convert DataFrame to list of dicts and bulk insert
        records = raw_df.to_dict(orient="records")
        helpers.bulk(es, [
            {"_index": ES_INDEX_OUTPUT, "_source": record}
            for record in records
        ])
        print(f"✅ Output {len(records)} records to: {ES_INDEX_OUTPUT}")
    except Exception as e:
        print(f"❌ Failed to predict or index: {e}")

# === Start Real-Time ML Monitoring Loop ===
print("🚀 Starting real-time SIEMVision ML inference...")
while True:
    logs_df = fetch_logs()  # Get latest logs
    raw_df, X = preprocess(logs_df)  # Preprocess the logs
    predict_and_output(raw_df, X)  # Run model predictions and output
    time.sleep(FETCH_INTERVAL)  # Wait for next interval
