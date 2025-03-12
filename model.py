import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib

# Load dataset
data = pd.read_excel("data/house_prices.xlsx")  # Load data from Excel

# Apply log transformation to the target variable (price)
y = np.log(data['price'])  # This reduces the skewness of the price distribution

# Select features
X = data[['area', 'bedrooms', 'bathrooms', 'location']]

# Step 1: Scale numerical features
scaler = StandardScaler()
X[['area', 'bedrooms', 'bathrooms']] = scaler.fit_transform(X[['area', 'bedrooms', 'bathrooms']])

# Save the scaler for future use
joblib.dump(scaler, "models/scaler.pkl")

# Step 2: One-hot encode categorical data (location)
X = pd.get_dummies(X, drop_first=True)

# Save column names before training (to match input during prediction)
joblib.dump(X.columns, "models/model_columns.pkl")

# Step 3: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the model
model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, "models/house_price_model.pkl")

print("Model trained and saved successfully!")
