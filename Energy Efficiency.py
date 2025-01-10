import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

# Load the dataset (replace 'energy_efficiency.csv' with your dataset file)
data = pd.read_csv('energy_efficiency.csv')

# Display the first few rows of the dataset
print(data.head())

# Feature engineering
# Add a heating-to-floor-area ratio (assuming 'Heating Load' and 'Floor Area' are columns in the dataset)
data['Heating_to_Floor_Area'] = data['Heating Load'] / data['Floor Area']

# Select features and target variable (assuming 'Energy Efficiency Rating' is the target column)
X = data[['Insulation Quality', 'Heating_to_Floor_Area', 'Orientation', 'Glazing Area']]
y = data['Energy Efficiency Rating']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the XGBoost model
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)

# Train the model
xgb_model.fit(X_train, y_train)

# Predict on the test set
y_pred = xgb_model.predict(X_test)

# Evaluate the model
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"R² Score: {r2:.2f}")
print(f"RMSE: {rmse:.2f}")

# Feature importance visualization
plt.figure(figsize=(10, 6))
plt.barh(X.columns, xgb_model.feature_importances_)
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Feature Importance in Predicting Energy Efficiency")
plt.show()

# Insights for homeowners
insights = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
insights['Difference'] = insights['Actual'] - insights['Predicted']
print("Actionable Insights:")
print(insights.head())

# Save actionable insights for further review
insights.to_csv('actionable_insights.csv', index=False)
