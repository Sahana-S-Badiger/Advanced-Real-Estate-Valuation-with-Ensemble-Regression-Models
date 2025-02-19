# Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Hyperparameter Tuning with GridSearchCV (Optional)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
}
grid_search_rf = GridSearchCV(estimator=RandomForestRegressor(random_state=42), param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search_rf.fit(X_train, y_train)
print(f"Best parameters for Random Forest: {grid_search_rf.best_params_}")
rf_model = grid_search_rf.best_estimator_

grid_search_gb = GridSearchCV(estimator=GradientBoostingRegressor(random_state=42), param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search_gb.fit(X_train, y_train)
print(f"Best parameters for Gradient Boosting: {grid_search_gb.best_params_}")
gb_model = grid_search_gb.best_estimator_

# Step 7: Model Evaluation
# Evaluate Random Forest
rf_predictions = rf_model.predict(X_test)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))
rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_r2 = r2_score(y_test, rf_predictions)

print("\nRandom Forest Results:")
print(f"RMSE: {rf_rmse}")
print(f"MAE: {rf_mae}")
print(f"R²: {rf_r2}")

# Evaluate Gradient Boosting
gb_predictions = gb_model.predict(X_test)
gb_rmse = np.sqrt(mean_squared_error(y_test, gb_predictions))
gb_mae = mean_absolute_error(y_test, gb_predictions)
gb_r2 = r2_score(y_test, gb_predictions)

print("\nGradient Boosting Results:")
print(f"RMSE: {gb_rmse}")
print(f"MAE: {gb_mae}")
print(f"R²: {gb_r2}")

# Step 8: Visualize Model Performance
# Model Comparison Plot
plt.figure(figsize=(10, 5))
plt.bar(['Random Forest', 'Gradient Boosting'], [rf_r2, gb_r2], color=['blue', 'green'])
plt.title('Model R² Comparison')
plt.ylabel('R² Score')
plt.show()

# Step 9: Additional Visualizations
# Pair Plot (to visualize pairwise relationships between features)
sns.pairplot(data)
plt.suptitle("Pair Plot of the Dataset", y=1.02)
plt.show()

# Box Plot (to check the distribution and detect outliers)
plt.figure(figsize=(12, 6))
sns.boxplot(data=data)
plt.title("Box Plot of the Dataset")
plt.xticks(rotation=90)
plt.show()

# Correlation Heatmap (to show correlation between numerical features)
numeric_data = data.select_dtypes(include=[np.number])
plt.figure(figsize=(12, 8))
correlation_matrix = numeric_data.corr()  # Compute correlation matrix
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()


# Residual Plot (to check residuals for regression models)
residuals_rf = y_test - rf_predictions
plt.figure(figsize=(10, 6))
sns.residplot(x=rf_predictions, y=residuals_rf, lowess=True, color="blue")
plt.title("Residual Plot (Random Forest)")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.show()

# Step 10: Save Models
joblib.dump(rf_model, 'rf_model.pkl')
joblib.dump(gb_model, 'gb_model.pkl')

print("\nModels saved as 'rf_model.pkl' and 'gb_model.pkl'")
