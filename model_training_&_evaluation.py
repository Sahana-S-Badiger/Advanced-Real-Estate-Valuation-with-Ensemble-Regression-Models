# Advanced Real Estate Valuation With Ensemble Regression Models
# Accurate Property Assessment Techniques

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
data_path = '/content/Real_Estate.csv'  # Adjust this path if necessary
data = pd.read_csv(data_path)

# Display dataset columns to verify the correct names
print("Columns in the dataset:", data.columns)

# Step 3: Define Target and Features
target_column = 'House price of unit area'  # Adjusted the column name

# Check if the target column exists in the dataset
if target_column not in data.columns:
    raise ValueError(f"Target column '{target_column}' not found in dataset. Please verify the column names.")

# Separate features (X) and target (y)
X = data.drop(columns=[target_column])
y = data[target_column]

# Step 4: Data Preprocessing
# Check for missing values
print("Missing values per column before processing:\n", data.isnull().sum())
data = data.dropna()  # Drop missing values (customize as needed)

# Convert categorical variables to dummy/indicator variables
categorical_cols = X.select_dtypes(include=['object']).columns
if len(categorical_cols) > 0:
    print(f"Categorical columns detected: {categorical_cols}")
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
