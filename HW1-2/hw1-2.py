import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# 1. 讀取 CSV 資料
file_path = r"C:\Users\jing5\Documents\HW1-2\2330-training.csv"
data = pd.read_csv(file_path)

# 2. 資料處理
# 轉換 Date 欄位為日期格式
data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y')
# 移除 y 欄位中的逗號並轉換為 float
data['y'] = data['y'].str.replace(',', '').astype(float)

# 重命名欄位以符合 Prophet 的要求
data.rename(columns={'Date': 'ds', 'y': 'y'}, inplace=True)

# 3. 使用 Prophet 模型預測
model = Prophet(changepoint_prior_scale=0.5, yearly_seasonality=False)
model.add_seasonality(name='monthly', period=30.5, fourier_order=5)

# 擬合模型
model.fit(data)

# 進行未來 60 天的預測
future = model.make_future_dataframe(periods=60)
forecast = model.predict(future)

# 4. 繪製預測結果圖
plt.figure(figsize=(12, 6))

# 實際數據（黑色線條）
plt.plot(data['ds'], data['y'], color='black', label='Actual Data')

# 預測數據（藍色線條）
plt.plot(forecast['ds'], forecast['yhat'], color='blue', label='Forecast')

# 不確定性區間（淺藍色陰影）
plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='lightblue', alpha=0.5)

# 歷史平均
historical_average = data['y'].mean()
plt.axhline(y=historical_average, color='blue', linestyle='-', label='Historical Average')

# 預測初始化點
plt.axvline(x=data['ds'].iloc[-1], color='red', linestyle='--', label='Forecast Initialization')

# 添加標題與標籤
plt.title('Stock Price Forecast')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.grid()

# 顯示圖表
plt.show()