import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import *
from keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
plt.style.use('ggplot') #グラフスタイル
plt.rcParams['figure.figsize'] = [12, 9] # グラフサイズ

# データセット読み込み
url = 'https://www.salesanalytics.co.jp/bgr8'
df = pd.read_csv(url)
# データ確認

# 変換
dataset = df['y'].values #NumPy配列へ変換
dataset = dataset.astype('float32')    #実数型へ変換
dataset = np.reshape(dataset, (-1, 1)) #1次元配列を2次元配列へ変換
print(dataset)

# ラグ付きデータセット生成関数
def gen_dataset(dataset, lag_max):
    X, y = [], []
    for i in range(len(dataset) - lag_max):
        a = i + lag_max
        X.append(dataset[i:a, 0]) #ラグ変数
        y.append(dataset[a, 0])   #目的変数
    return np.array(X), np.array(y)

# ラグ付きデータセットの生成
lag_max = 30
X, y = gen_dataset(dataset, lag_max)
print('X:',X.shape) #確認
print('y:',y.shape) #確認
