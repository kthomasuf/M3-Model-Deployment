from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
from tensorflow import keras
import json
import requests
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)

models = {
    'linear_regression': joblib.load('models/linear_regression.pkl'),
    'random_forest': joblib.load('models/random_forest.pkl'),
}

feature_columns = [
    "aqi_mean", "aqi_p90", "aqi_max", "days_reported",
    "year", "month", "week_of_year", "quarter",
    "aqi_mean_lag1", "aqi_mean_lag2",
    "aqi_max_lag1", "aqi_max_lag2",
    "aqi_p90_lag1", "aqi_p90_lag2",
    "aqi_mean_rolling_3"
]

@app.route('/')
def tool():
    return render_template('tool.html')

@app.route('/visualizations')
def visualizations():
    return render_template('visualizations.html')

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)
    return img_base64

model_df = pd.read_csv("csv/modeling_dataset.csv")
pred_df = pd.read_csv("csv/model_predictions.csv")
feature_importance = pd.read_csv("csv/feature_importance.csv")

model_df["week"] = pd.to_datetime(model_df["week"])
pred_df["week"] = pd.to_datetime(pred_df["week"])

@app.route('/charts', methods=['POST'])
def charts():
    try:
        data = request.get_json()
        state = data['state']
        model_col = data['model']
        aqi_metric = data['aqi_metric']

        state_pred = pred_df[pred_df["state"] == state].copy()
        state_model = model_df[model_df["state"] == state].copy()

        fig1, ax1 = plt.subplots()
        ax1.plot(state_pred["week"], state_pred["total_respiratory_admissions"], label="Actual")
        ax1.plot(state_pred["week"], state_pred[model_col], label="Predicted")
        ax1.set_title(f"{state}: Actual vs Predicted")
        ax1.set_xlabel("Week")
        ax1.set_ylabel("Admissions")
        ax1.legend()
        plt.xticks(rotation=45)

        fig2, ax2 = plt.subplots()
        ax2.scatter(state_model[aqi_metric], state_model["total_respiratory_admissions"])
        ax2.set_title(f"{state}: {aqi_metric} vs Admissions")
        ax2.set_xlabel(aqi_metric)
        ax2.set_ylabel("Total Respiratory Admissions")

        fig3, ax3 = plt.subplots()
        ax3.plot(state_model["week"], state_model[aqi_metric])
        ax3.set_title(f"{state}: {aqi_metric} Over Time")
        ax3.set_xlabel("Week")
        ax3.set_ylabel("AQI")
        plt.xticks(rotation=45)

        fig4, ax4 = plt.subplots()
        top_features = feature_importance.head(10)
        ax4.barh(top_features["feature"], top_features["importance"])
        ax4.invert_yaxis()
        ax4.set_title("Top 10 Feature Importances")
        ax4.set_xlabel("Importance")

        return jsonify({
            'actual_vs_predicted': fig_to_base64(fig1),
            'scatter':             fig_to_base64(fig2),
            'aqi_over_time':       fig_to_base64(fig3),
            'feature_importance':  fig_to_base64(fig4),
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def to_float(val):
    try:
        return float(val)
    except (ValueError, TypeError):
        return float('nan')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    f = data['features']

    raw = pd.DataFrame([{
        'aqi_w1_1': to_float(f['aqi_w1_1']),
        'aqi_w1_2': to_float(f['aqi_w1_2']),
        'aqi_w1_3': to_float(f['aqi_w1_3']),
        'aqi_w1_4': to_float(f['aqi_w1_4']),
        'aqi_w1_5': to_float(f['aqi_w1_5']),
        'aqi_w1_6': to_float(f['aqi_w1_6']),
        'aqi_w1_7': to_float(f['aqi_w1_7']),
        'aqi_w2_1': to_float(f['aqi_w2_1']),
        'aqi_w2_2': to_float(f['aqi_w2_2']),
        'aqi_w2_3': to_float(f['aqi_w2_3']),
        'aqi_w2_4': to_float(f['aqi_w2_4']),
        'aqi_w2_5': to_float(f['aqi_w2_5']),
        'aqi_w2_6': to_float(f['aqi_w2_6']),
        'aqi_w2_7': to_float(f['aqi_w2_7']),
        'aqi_w3_1': to_float(f['aqi_w3_1']),
        'aqi_w3_2': to_float(f['aqi_w3_2']),
        'aqi_w3_3': to_float(f['aqi_w3_3']),
        'aqi_w3_4': to_float(f['aqi_w3_4']),
        'aqi_w3_5': to_float(f['aqi_w3_5']),
        'aqi_w3_6': to_float(f['aqi_w3_6']),
        'aqi_w3_7': to_float(f['aqi_w3_7']),
    }])

    date = pd.to_datetime(f['date'])

    w1 = raw[[f'aqi_w1_{i}' for i in range(1, 8)]].values.flatten()
    w2 = raw[[f'aqi_w2_{i}' for i in range(1, 8)]].values.flatten()
    w3 = raw[[f'aqi_w3_{i}' for i in range(1, 8)]].values.flatten()

    w1 = w1[~np.isnan(w1)]
    w2 = w2[~np.isnan(w2)]
    w3 = w3[~np.isnan(w3)]

    model_input = pd.DataFrame([{
        'year':               date.year,
        'month':              date.month,
        'week_of_year':       date.isocalendar().week,
        'quarter':            date.quarter,
        'days_reported':      len(w1) + len(w2) + len(w3),

        'aqi_mean':           float(w1.mean()),
        'aqi_max':            float(w1.max()),
        'aqi_p90':            float(pd.Series(w1).quantile(0.90)),

        'aqi_mean_lag1':      float(w2.mean()),
        'aqi_max_lag1':       float(w2.max()),
        'aqi_p90_lag1':       float(pd.Series(w2).quantile(0.90)),

        'aqi_mean_lag2':      float(w3.mean()),
        'aqi_max_lag2':       float(w3.max()),
        'aqi_p90_lag2':       float(pd.Series(w3).quantile(0.90)),

        'aqi_mean_rolling_3': float(np.mean([w3.mean(), w2.mean(), w1.mean()])),
    }])

    model_input = model_input.reindex(columns=feature_columns, fill_value=0)

    model = models['random_forest']
    total_cases = model.predict(model_input)[0]

    prompt = f""" You are an advisor helping explain total expected respiratory hospitalizations for a given area.
    Do NOT use alarming or urgent langauge. Do not roleplay as a medical professional.
    
    In 2 to 3 sentences MAX, explain how many total respiratory hospitalizations are expected for the user's area and
    what they can do to prepare their community.

    Total respiratory hospitalizations expected for this area: {total_cases}
    """

    llm_response = call_ollama(prompt)

    return jsonify({
        'total_cases': round(total_cases, 0),
        'llm_response': llm_response
    })

def call_ollama(prompt):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "phi3",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": 200,
                    "temperature": 0.1
                }
            },
            timeout=120
        )
        if response.status_code == 200:
            return response.json()["response"].strip()
        else:
            return "Unable to generate response at this time."
    except requests.exceptions.ConnectionError:
        return "Service unavailable. Please ensure Ollama is running."
    except requests.exceptions.Timeout:
        return "Request timed out. Please try again."

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)