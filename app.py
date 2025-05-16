from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load trained model & encoders
model = joblib.load("real_estate_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# Load real estate data for recommendations
df = pd.read_excel(r"C:\Users\91846\Desktop\b\real_estate_bhopal.xlsx")


@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input
        location = request.form['location']
        property_type = request.form['property_type']
        size_sqft = float(request.form['size_sqft'])
        bedrooms = int(request.form['bedrooms'])
        bathrooms = int(request.form['bathrooms'])
        balconies = int(request.form['balconies'])
        furnishing_status = request.form['furnishing_status']

        # Encode categorical values
        location_encoded = label_encoders['location'].transform([location])[0]
        property_type_encoded = label_encoders['property_type'].transform([property_type])[0]
        furnishing_encoded = label_encoders['furnishing_status'].transform([furnishing_status])[0]

        # Prepare input for model
        input_data = np.array([[location_encoded, property_type_encoded, size_sqft, bedrooms, bathrooms, balconies, furnishing_encoded]])

        # Predict price
        predicted_price = model.predict(input_data)[0]

        # Find similar properties from dataset
        similar_properties = df[
            (df['location'] == location) &
            (df['size_sqft'].between(size_sqft - 200, size_sqft + 200))  # Size approx match
        ].head(3)  # Top 3 matches

        # Convert to JSON format
        recommendations = similar_properties[['property_type', 'size_sqft', 'price', 'location']].to_dict(orient='records')

        # Return result
        return jsonify({
            "Predicted Price": f"₹{round(predicted_price, 2)}",
            "Similar Properties": recommendations
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)