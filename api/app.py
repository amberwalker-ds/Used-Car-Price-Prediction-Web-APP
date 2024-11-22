from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import xgboost as xgb
from data_process_live import create_is_luxury_column, add_luxury_and_popularity_features
import scripts.config

# Load the saved preprocessing pipeline
pipeline = joblib.load(scripts.config.pipeline_save_path)

# Load the trained XGBoost model
xgb_model = xgb.XGBRegressor()
xgb_model.load_model(scripts.config.model_save_path)

# Replace the model in the pipeline with the loaded XGBoost model
pipeline.named_steps['xgb'].regressor_ = xgb_model

# Initialize Flask app
app = Flask(__name__)

# Helper function to preprocess live data
def preprocess_live_data(X):
    """
    Preprocess live data to include all features required by the model.
    """
    X = X.copy()
    current_year = datetime.year

    bins = [0, 50000, 100000, 150000, np.inf]
    labels = ['low', 'medium', 'high', 'very high']

    # Car Age
    X['car_age'] = current_year - X['year']

    # Handle missing listing_month
    current_month = datetime.now().month
    X['listing_month'] = X['listing_month'].fillna(current_month)

    #listingyear
    X['listing_year'] = current_year
    X = create_is_luxury_column(X, luxury_brands, luxury_models)
    # Mileage Bins
    X['mileage_bin'] = pd.cut(X['mileage'], bins=bins, labels=labels, right=False)

    # Fuel Type One-Hot Encoding
    fuel_types = ['bifuel', 'diesel', 'hybrid', 'petrol']
    for fuel in fuel_types:
        X[f'fuel_{fuel}'] = (X['fuel_type'].str.lower() == fuel).astype(int)

    # Gearbox Type
    X['gearbox_type_automatic'] = (X['gear_type'].str.lower() == 'automatic').astype(int)

    X = add_luxury_and_popularity_features(X)
    X = pd.get_dummies(X, columns=['country_of_origin'], prefix='country').copy()
    X['log_mileage'] = np.log(X['mileage'] + 1)
    X['luxury_age_interaction'] = X['is_luxury'] * X['car_age']
    X['age_auto_interaction'] = X['gearbox_type_automatic'] * X['car_age']
    X['age_engine_interaction'] = X['engine'] * X['car_age']
    X['lux_auto_interaction'] = X['gearbox_type_automatic'] * X['is_luxury']
    #interaction term for age and mileage
    X['age_mileage_interaction'] = X['car_age'] * X['log_mileage']

    return X


@app.route('/predict', methods=['POST'])
def predict_single():
    """
    Endpoint for single-row prediction.
    """
    data = request.json  # Expecting JSON input
    try:
        # Convert input to DataFrame
        df = pd.DataFrame([data])
        # Preprocess the live data
        processed_data = preprocess_live_data(df)
        # Predict
        prediction = pipeline.predict(processed_data)
        return jsonify({'predicted_price': prediction[0]})
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


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)