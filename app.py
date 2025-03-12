from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np  # Make sure numpy is imported for the np.exp function

app = Flask(__name__)

# Load trained model and column names
model = joblib.load("models/house_price_model.pkl")  # Load trained model
model_columns = joblib.load("models/model_columns.pkl")  # Load feature columns
scaler = joblib.load("models/scaler.pkl")  # Load the scaler

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input values from the form
        area = int(request.form["area"])
        bedrooms = int(request.form["bedrooms"])
        bathrooms = int(request.form["bathrooms"])
        location = request.form["location"]

        # Create DataFrame with input values
        input_features = pd.DataFrame([[area, bedrooms, bathrooms, location]], 
                                      columns=['area', 'bedrooms', 'bathrooms', 'location'])

        # Scale the numerical features using the loaded scaler
        input_features[['area', 'bedrooms', 'bathrooms']] = scaler.transform(input_features[['area', 'bedrooms', 'bathrooms']])

        # One-hot encode the location (same as during training)
        input_features = pd.get_dummies(input_features, drop_first=True)

        # Ensure input matches model features (add missing columns and set their values to 0)
        input_features = input_features.reindex(columns=model_columns, fill_value=0)

        # Predict price (log-transformed)
        predicted_log_price = model.predict(input_features)[0]

        # Reverse the log transformation to get the actual price
        predicted_price = np.exp(predicted_log_price)

        # Return the predicted price to the user
        return render_template("result.html", price=round(predicted_price, 2))

    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)
