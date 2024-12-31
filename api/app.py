from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import xgboost as xgb
import os
from data_process_live import create_is_luxury_column, add_luxury_and_popularity_features
import config
from flask_cors import CORS
from google.cloud import storage
import joblib
import xgboost as xgb
import os
from google.cloud import storage

def download_model(bucket_name, source_blob_name, local_path):
    """
    Fetches a model file from Google Cloud Storage and stores it locally.

    :param bucket_name: Name of the Google Cloud Storage bucket
    :param source_blob_name: Name of the file in the bucket
    :param local_path: Path where the file should be saved locally
    """
    # Ensure the local directory exists
    local_directory = os.path.dirname(local_path)
    if not os.path.exists(local_directory):
        os.makedirs(local_directory)

    # Initialize the Google Cloud Storage client
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    # Download the file
    blob.download_to_filename(local_path)
    print(f"Model {source_blob_name} downloaded to {local_path}.")

# Download models from Cloud Storage
if not os.path.exists("models/saved_pipeline.pkl"):
    download_model("car-price-prediction-models", "models/saved_pipeline.pkl", "models/saved_pipeline.pkl")


if not os.path.exists("models/saved_xgb_model.json"):
    download_model("car-price-prediction-models", "models/saved_xgb_model.json", "models/saved_xgb_model.json")

# Load models
pipeline = joblib.load(config.pipeline_save_path)
xgb_model = xgb.XGBRegressor()
xgb_model.load_model(config.model_save_path)

# Replace the model in the pipeline with the loaded XGBoost model
pipeline.named_steps['xgb'].regressor_ = xgb_model

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "https://datawithamber.com"}})

# Helper function to preprocess live data
def preprocess_live_data(X):
    """
    Preprocess live data to include all features required by the model.
    """
    X = X.copy()

    for col in X.select_dtypes(include="object").columns:
      X[col] = X[col].str.lower().str.replace(" ", "")

    for col in ['year', 'mileage', 'engine']:
      X[col] = pd.to_numeric(X[col], errors='coerce')

    current_year = datetime.now().year
    X['listing_year'] = current_year
    X = create_is_luxury_column(X)
    X['car_age'] = current_year - X['year']
    current_month = datetime.now().month
    if 'listing_month' not in X.columns:
        X['listing_month'] = current_month  # Default to the current month if missing
    else:
        X['listing_month'] = X['listing_month'].fillna(current_month)

    fuel_types = ['bifuel', 'diesel', 'hybrid', 'petrol']
    for fuel in fuel_types:
        X[f'fuel_{fuel}'] = 0  # Initialize columns with 0
    if 'fuel_type' in X.columns:
        for fuel in fuel_types:
            X[f'fuel_{fuel}'] = (X['fuel_type'] == fuel).astype(int)

    country_columns = [
        'country_germany', 'country_japan', 'country_spain', 'country_france', 'country_ukraine',
        'country_russia', 'country_sweden', 'country_italy', 'country_czech republic',
        'country_unknown', 'country_uk', 'country_romania', 'country_usa', 'country_south korea'
    ]
    for col in country_columns:
        X[col] = 0  # Initialize columns with 0

    X = add_luxury_and_popularity_features(X)

    if 'country_of_origin' in X.columns:
        X = pd.get_dummies(X, columns=['country_of_origin'], prefix='country')
        for col in country_columns:
            if col not in X.columns:
                X[col] = 0  # Ensure all expected columns are present
    
    X['gearbox_type_automatic'] = (X['gear_type'].str.lower() == 'automatic').astype(int)
    X['log_mileage'] = np.log(X['mileage'] + 1)
    X['luxury_age_interaction'] = X['is_luxury'] * X['car_age']
    X['age_engine_interaction'] = X['engine'] * X['car_age']
    X['lux_auto_interaction'] = X['gearbox_type_automatic'] * X['is_luxury']
    X['age_mileage_interaction'] = X['car_age'] * X['log_mileage']

    return X  

@app.route('/predict', methods=['POST'])
def predict_single():
    """
    Endpoint for single-row prediction.
    """
    try:
        #fetch token
        data = request.json  # Expecting JSON input
        # Convert input to DataFrame
        df = pd.DataFrame([data])
        # Preprocess the live data
        processed_data = preprocess_live_data(df)
        # Predict
        # Align processed data with the expected columns of the model
        expected_columns = pipeline.named_steps['preprocessor'].transformers_[0][1].get_feature_names_out()
        processed_data = processed_data.reindex(columns=expected_columns, fill_value=0)
        
        prediction = pipeline.predict(processed_data)
        # Convert prediction to Python float for JSON serialization
        predicted_price = round(float(prediction[0]),2)
        return jsonify({'predicted_price': predicted_price})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/bulk-predict', methods=['POST'])
def predict_bulk():
    """
    Endpoint for bulk predictions from a CSV file.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Read the CSV file into a DataFrame
        data = pd.read_csv(file)
        # Preprocess the live data
        processed_data = preprocess_live_data(data)
        # Predict
        predictions = pipeline.predict(processed_data)
        # Add predictions as a new column
        data['predicted_price'] = predictions
        # Convert the result to JSON
        result = data.to_json(orient='records')
        return result
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))  # Default to 8080 if PORT is not set
    app.run(host="0.0.0.0", port=port)