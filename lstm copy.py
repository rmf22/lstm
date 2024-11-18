# -*- coding: utf_8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LSTM, Dense

from dataset import stock



# ========== データの規格化 ==========
data = stock['Close'] # 特徴量として「終値」を選択
scalar = MinMaxScaler(feature_range = (0, 1))
np_data = np.array(data).reshape(-1, 1)
scaled_data = scalar.fit_transform(np_data) # 規格化



# ========== 訓練データとテストデータの準備 ==========
p = 70  # 総データのうち, 1番初めからp%を訓練データに使用
train_size = int(len(data) * (p / 100))
train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]  # 全データを訓練データとテストデータに分割


def create_dataset(data, time_step = 1):
    X, Y = [], []   # 説明変数, 目的変数

    for i in range(len(data) - time_step - 1):
        a = i + time_step
        X.append(data[i:a, 0])  # 説明変数としてのラグ変数
        Y.append(data[a, 0])    # 目的変数

    return np.array(X), np.array(Y)


time_step = 20  # タイムステップ数

X_train, Y_train = create_dataset(train_data, time_step)
X_test, Y_test = create_dataset(test_data, time_step)

# LSTMの入力の形状を (sample(サンプル数), time_steps(タイムステップ), features(特徴量数)) に変換
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)



# ========== LSTMモデル構築と学習 ==========
input_dim = 1           # 入力データの次元数
output_dim = 1          # 出力データの次元数
hidden_layers = 50      # 隠れ層のユニット数
batch_size = 128        # ミニバッチサイズ
epochs = 10            # 学習エポック数
learning_rate = 0.001   # 学習率

model = Sequential()
model.add(LSTM(hidden_layers, return_sequences = False, input_shape = (time_step, input_dim)))   # 1層目
#model.add(LSTM(hidden_layers, return_sequences = False))    # 2層目
#model.add(Dense(25))
model.add(Dense(output_dim))

optimizer = Adam(lr = learning_rate)
model.compile(optimizer = optimizer, loss = 'mean_squared_error')
model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs)

# テストデータの予測
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# 逆スケーリング
train_predict = scalar.inverse_transform(train_predict)
test_predict = scalar.inverse_transform(test_predict)
Y_test = scalar.inverse_transform([Y_test])

MAE = mean_absolute_error(Y_test[0], test_predict[:, 0])     # 二乗平均誤差
R2 = r2_score(Y_test[0], test_predict[:, 0])                # 決定係数

print(f'MSE: {MAE: .4f}')
print(f'R^2: {R2: .4f}')



# ========== 結果のプロット ==========
plt.figure(figsize = (10, 6))
plt.title(f'NTT Stock Price Prediction: p = {p}, time_span = {time_step}')
plt.plot(data.index, data, label = 'Historical')
plt.text(0.5, 0.5, f'MSE: {MAE: .4f}, R^2: {R2: .4f}', fontsize = 12, transform = plt.transAxes)

#train_predict_plot = np.empty_like(scaled_data)
#train_predict_plot[:, :] = np.nan
#train_predict_plot[time_step: len(train_predict) + time_step, :]
#plt.plot(data.index, train_predict_plot, label = 'Train Predict')

test_predict_plot = np.empty_like(scaled_data)
test_predict_plot[:, :] = np.nan
test_predict_plot[len(train_predict) + time_step * 2 + 1:len(scaled_data) - 1, :] = test_predict
plt.plot(data.index, test_predict_plot, label = 'Test Predict')

plt.xlabel('Date')
plt.ylabel('Price (yen)')
plt.legend()
plt.show()
