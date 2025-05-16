import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib

# Sample dataset load (replace with actual dataset)
df = pd.read_excel(r"C:\Users\91846\Desktop\b\real_estate_bhopal.xlsx")



# Encode categorical variables
label_encoders = {}
for col in ['location', 'property_type', 'furnishing_status']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Features & Target
X = df[['location', 'property_type', 'size_sqft', 'bedrooms', 'bathrooms', 'balconies', 'furnishing_status']]
y = df['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, "real_estate_model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")