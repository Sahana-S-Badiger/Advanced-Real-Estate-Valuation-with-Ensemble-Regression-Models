from django.shortcuts import render
import os
import joblib
import numpy as np
import pandas as pd

# Load pre-trained model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, 'prediction', 'models', 'rf_model.pkl')

try:
    model = joblib.load(model_path)
except FileNotFoundError:
    model = None
    print("Model file 'rf_model.pkl' not found. Please check the file path.")

# Load feature names (column names) used during training
columns_path = os.path.join(BASE_DIR, 'prediction', 'models', 'columns.pkl')

try:
    expected_columns = joblib.load(columns_path)  # Load saved feature names
except Exception as e:
    print("Columns file not found. Ensure columns.pkl exists:", e)
    expected_columns = None

def home(request):
    """
    View to render the home page with the predict button.
    """
    return render(request, 'prediction/home.html')

def predict_price(request):
    """
    View to handle house price prediction.
    """
    if request.method == 'POST':
        try:
            # Extract input values from POST request
            inputs = {
                'Transaction Date': float(request.POST['transaction_date']),
                'House Age': float(request.POST['house_age']),
                'Distance to the nearest MRT station': float(request.POST['distance_to_mrt']),
                'Number of convenience stores': int(request.POST['convenience_stores']),
                'Latitude': float(request.POST['latitude']),
                'Longitude': float(request.POST['longitude']),
            }

            # Convert input into a DataFrame
            input_df = pd.DataFrame([inputs])

            # Check expected_columns and align input_df
            if expected_columns is not None and len(expected_columns) > 0:
                input_df = input_df.reindex(columns=expected_columns, fill_value=0)
            else:
                raise ValueError("Expected columns are not properly loaded or are empty.")

            # Check if the model is loaded
            if model is None:
                raise Exception("Model file not loaded. Ensure the 'rf_model.pkl' file is in the correct location.")

            # Make prediction
            predicted_price = model.predict(input_df)[0]

            # Pass results to the result template
            context = {
                'predicted_price': round(predicted_price, 2),
                'inputs': inputs,
            }
            return render(request, 'prediction/result.html', context)

        except Exception as e:
            # Pass error message and inputs back to the prediction form
            return render(request, 'prediction/predict.html', {
                'error': f"Error occurred: {str(e)}",
                'inputs': request.POST,
            })

    # Render the prediction form template for GET requests
    return render(request, 'prediction/predict.html')
