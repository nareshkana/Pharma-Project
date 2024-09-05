from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import os
import sys
from flask import Flask, jsonify, request
# from flask_cors import CORS
# import pickle
import pandas as pd
# import pdarima




app = Flask(__name__)



# Routes for rendering pages
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/forecasting')
def forecasting():
    return render_template('forecasting.html')

@app.route('/drug-info')
def drug():
    return render_template('drug.html')

@app.route('/aboutus')
def aboutus():
    return render_template('aboutus.html')

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

@app.route('/prediction2')
def prediction():
    return render_template('predictions.html')

@app.route('/search')
def search():
    return render_template('search.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/no5c')
def no5c():
    return render_template('no5c.html')

@app.route('/no5b')
def no5b():
    return render_template('no5b.html')

@app.route('/mo1ab')
def mo1ab():
    return render_template('mo1ab.html')

@app.route('/mo1ae')
def mo1ae():
    return render_template('mo1ae.html')

@app.route('/no2ba')
def no2ba():
    return render_template('no2ba.html')

@app.route('/statistics')
def statistics():
    return render_template('statistics.html')

@app.route('/details')
def details():
    return render_template('detail.html')

@app.route('/drugs')
def drugs():
    return render_template('drugs.html')
# Path to the models folder
models_folder = 'models'

@app.route('/prediction')
def prediction2():
    return render_template('prediction2.html')

# Load models for weekly and daily predictions
drugs = ['M01AB', 'M01AE', 'N02BA', 'N02BE', 'N05B', 'N05C', 'R03', 'R06']

models_weekly = {}
models_daily = {}

for drug in drugs:
    weekly_model_path = os.path.join(models_folder, f'auto_arima_model_Week_{drug}.pkl')
    daily_model_path = os.path.join(models_folder, f'auto_arima_model_{drug}.pkl')

    # Ensure the file exists before trying to load it
    if os.path.exists(weekly_model_path):
        with open(weekly_model_path, 'rb') as file:
            models_weekly[drug] = pickle.load(file)
    else:
        print(f"Weekly model for {drug} not found.")

    if os.path.exists(daily_model_path):
        with open(daily_model_path, 'rb') as file:
            models_daily[drug] = pickle.load(file)
    else:
        print(f"Daily model for {drug} not found.")



@app.route('/submit', methods=['POST'])
def submit():
    date_type = request.form.get('dateType')

    result = {}
    if date_type == 'daily':
        single_date = request.form.get('singleDate')
        if single_date:
            result = predict_daily_sales(single_date)
    elif date_type == 'weekly':
        start_date = request.form.get('startDate')
        end_date = request.form.get('endDate')
        if start_date and end_date:
            result = predict_weekly_sales(start_date, end_date)
    
    return jsonify(result)

def predict_daily_sales(single_date):
    start_date = '2019-11-30'  # Last date in the dataset
    date_range = pd.date_range(start=start_date, end=single_date, freq='D')

    predictions = {}
    for drug in drugs:
        if drug in models_daily:
            predictions[drug] = models_daily[drug].predict(n_periods=len(date_range)) / 30

    predictions_df = pd.DataFrame(predictions, index=date_range)
    if single_date in predictions_df.index:
        result = predictions_df.loc[single_date].to_dict()
    else:
        result = {"error": "Date out of prediction range"}
    return result

def predict_weekly_sales(start_date, end_date):
    date_range = pd.date_range(start=start_date, end=end_date, freq='7D')

    predictions = {}
    for drug in drugs:
        if drug in models_weekly:
            predictions[drug] = models_weekly[drug].predict(n_periods=len(date_range))

    predictions_df = pd.DataFrame(predictions, index=date_range)
    if end_date in predictions_df.index:
        result = predictions_df.loc[end_date].to_dict()
    else:
        result = {"error": "Date out of prediction range"}
    return result









if __name__ == "__main__":

    app.run(debug=True,port=5004)

 
