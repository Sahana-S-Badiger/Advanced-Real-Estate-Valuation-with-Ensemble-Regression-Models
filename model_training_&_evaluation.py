# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Step 2: Load Dataset
data_path = '/content/Real_Estate.csv'  # Update this path as needed
data = pd.read_csv(data_path)

# Display dataset columns to verify the correct names
print("Columns in the dataset:", data.columns)

# Step 3: Define Target and Features
target_column = 'House price of unit area'  # Update column name if needed

# Verify if the target column exists in the dataset
if target_column not in data.columns:
    raise ValueError(f"Target column '{target_column}' not found in dataset. Please verify the column names.")

# Separate features (X) and target (y)
X = data.drop(columns=[target_column])
y = data[target_column]

# Step 4: Data Preprocessing
# Check and handle missing values
print("Missing values per column before processing:\n", data.isnull().sum())
data = data.dropna()  # Drop missing values (modify as required)

from django.shortcuts import render
import os
import joblib
import numpy as np
import pandas as pd

# Load pre-trained model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(_file_)))
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

# Convert categorical variables into dummy variables
categorical_cols = X.select_dtypes(include=['object']).columns
if len(categorical_cols) > 0:
    print(f"Categorical columns detected: {categorical_cols}")
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Step 8: Visualizing Model Performance
# Compare R² scores of both models
plt.figure(figsize=(10, 5))
plt.bar(['Random Forest', 'Gradient Boosting'], [rf_r2, gb_r2], color=['blue', 'green'])
plt.title('Model R² Comparison')
plt.ylabel('R² Score')
plt.show()

# Step 9: Additional Data Visualizations
# Pair Plot
sns.pairplot(data)
plt.suptitle("Pair Plot of the Dataset", y=1.02)
plt.show()

# Box Plot to detect outliers
plt.figure(figsize=(12, 6))
sns.boxplot(data=data)
plt.title("Box Plot of the Dataset")
plt.xticks(rotation=90)
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12, 8))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

# Residual Plot for Random Forest
residuals_rf = y_test - rf_predictions
plt.figure(figsize=(10, 6))
sns.residplot(x=rf_predictions, y=residuals_rf, lowess=True, color="blue")
plt.title("Residual Plot (Random Forest)")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.show()

# Step 10: Save Trained Models
joblib.dump(rf_model, 'rf_model.pkl')
joblib.dump(gb_model, 'gb_model.pkl')

print("\nModels saved as 'rf_model.pkl' and 'gb_model.pkl'"
