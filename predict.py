import joblib
import pandas as pd

def load_artifacts():
    model = joblib.load('loan_model.pkl')
    scaler = joblib.load('scaler.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    return model, scaler, label_encoders

def predict_loan(input_data: dict):
    model, scaler, label_encoders = load_artifacts()
    
    df = pd.DataFrame([input_data])
    
    # Encode categorical features
    for col, le in label_encoders.items():
        if col in df.columns:
            val = str(df[col].iloc[0]).strip()
            df[col] = le.transform([val]) if val in le.classes_ else 0

    # Scale features
    X_scaled = scaler.transform(df)
    
    # Predict
    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0].max()
    
    status = "Approved" if prediction == 1 else "Rejected"
    return {"status": status, "confidence": float(probability)}

if __name__ == "__main__":
    # Sample Application for Testing
    sample_application = {
        'no_of_dependents': 2,
        'education': 'Graduate',
        'self_employed': 'No',
        'income_annum': 5000000,
        'loan_amount': 15000000,
        'loan_term': 36,
        'cibil_score': 750,
        'residential_assets_value': 2000000,
        'commercial_assets_value': 0,
        'luxury_assets_value': 500000,
        'bank_asset_value': 1000000
    }
    
    result = predict_loan(sample_application)
    print(f"Prediction Result: {result}")
