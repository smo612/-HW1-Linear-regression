import os
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for
from prophet import Prophet
import matplotlib.pyplot as plt

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', forecast_image=None)

@app.route('/predict', methods=['POST'])
def predict():
    # 確保上傳的檔案存在
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    # 儲存上傳的檔案
    file_path = os.path.join('static', file.filename)
    file.save(file_path)

    # 讀取 CSV 資料
    data = pd.read_csv(file_path)

    # 資料處理
    data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y')
    data['y'] = data['y'].str.replace(',', '').astype(float)
    data.rename(columns={'Date': 'ds', 'y': 'y'}, inplace=True)

    # 使用 Prophet 模型預測
    model = Prophet(changepoint_prior_scale=0.5, yearly_seasonality=False)
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    model.fit(data)

    future = model.make_future_dataframe(periods=60)
    forecast = model.predict(future)

    # 繪製預測結果圖
    plt.figure(figsize=(12, 6))
    plt.plot(data['ds'], data['y'], color='black', label='Actual Data')
    plt.plot(forecast['ds'], forecast['yhat'], color='blue', label='Forecast')
    plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='lightblue', alpha=0.5)
    historical_average = data['y'].mean()
    plt.axhline(y=historical_average, color='blue', linestyle='-', label='Historical Average')
    plt.axvline(x=data['ds'].iloc[-1], color='red', linestyle='--', label='Forecast Initialization')
    plt.title('Stock Price Forecast')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.grid()

    # 儲存圖表到 static 文件夾
    image_path = os.path.join('static', 'forecast.png')
    plt.savefig(image_path)
    plt.close()

    return render_template('index.html', forecast_image='forecast.png')

if __name__ == '__main__':
    app.run(debug=True)