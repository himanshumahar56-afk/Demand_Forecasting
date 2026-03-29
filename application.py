from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd
import datetime

application = Flask(__name__)
app = application

# Load model files
model = joblib.load("demand_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")  # trained feature order

@application.route('/')
def home():
    return render_template("index.html")

@application.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form.to_dict()
          # --- Check for empty inputs ---
        for key, value in data.items():
            if value.strip() == "":
                return render_template(
                    'index.html',
                    prediction_text=f"Please Fill The Entry: {key}"
                )
        # --- Convert inputs to float ---
        price = float(data['Price'])
        discount = float(data['Discount'])
        inventory = float(data['Inventory Level'])
        competitor = float(data['Competitor Pricing'])

        lag1 = float(data['lag_1_demand'])
        lag7 = float(data['lag_7_demand'])
        lag30 = float(data['lag_30_demand'])

        promotion = float(data['Promotion'])
        seasonality = float(data['Seasonality'])
        month = int(data['Month'])

        # --- Auto date features ---
        today = datetime.datetime.now()
        weekday = today.weekday()
        weekofyear = today.isocalendar()[1]
        quarter = (month - 1) // 3 + 1

        # --- Rolling means ---
        rolling_mean_7 = (lag1 + lag7) / 2
        rolling_mean_14 = (lag7 + lag30) / 2
        rolling_mean_30 = (lag1 + lag7 + lag30) / 3
        lag14 = (lag7 + lag30) / 2

        # --- Input DataFrame ---
        input_dict = {
            'Price': price,
            'Discount': discount,
            'Inventory Level': inventory,
            'Competitor Pricing': competitor,
            'Seasonality': seasonality,
            'Promotion': promotion,
            'Month': month,
            'Weekday': weekday,
            'Weekofyear': weekofyear,
            'Quarter': quarter,
            'lag_1_demand': lag1,
            'lag_7_demand': lag7,
            'lag_14_demand': lag14,
            'lag_30_demand': lag30,
            'Rolling_Mean_7': rolling_mean_7,
            'Rolling_Mean_14': rolling_mean_14,
            'Rolling_Mean_30': rolling_mean_30
        }

        input_df = pd.DataFrame([input_dict])

        # --- Add missing columns if any (only numeric features from training) ---
        for col in features:
            if col not in input_df.columns:
                input_df[col] = 0

        # --- Ensure correct feature order ---
        input_df = input_df[features]

        # --- Debug print ---
        print("Input DataFrame for model:\n", input_df)

        # --- Scale and predict ---
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)
        prediction = np.expm1(prediction)

        predicted_units = int(round(prediction[0]))

        return render_template(
            'index.html',
            prediction_text=f'{predicted_units} Units Expected Sales'
        )

    except Exception as e:
        return render_template(
            'index.html',
            prediction_text=f'Error: {str(e)}'
        )

if __name__ == "__main__":
    application.run(debug=True)