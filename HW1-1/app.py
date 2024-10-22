from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_data', methods=['POST'])
def generate_data():
    a = float(request.form['a'])
    noise = float(request.form['noise'])
    num_points = int(request.form['num_points'])
    b = 50

    x = np.random.uniform(-10, 10, num_points)
    y = a * x + b + noise * np.random.normal(0, 1, num_points)

    # 擬合線性回歸模型
    coefficients = np.polyfit(x, y, 1)
    y_fit = coefficients[0] * x + coefficients[1]

    # 繪製圖表
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color='blue', label='Data Points')
    plt.plot(x, y_fit, color='red', label='Regression Line')
    plt.title('Linear Regression')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    
    # 保存圖表為圖片
    img = io.BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    
    return jsonify({'plot_url': plot_url})

if __name__ == '__main__':
    app.run(debug=True)