import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

from dataset import stock

data = stock['Close']
plt.plot(data.index, data)
plt.title((f'NTT Stock Price (1988~2024)'))
plt.xlabel('Date')
plt.ylabel('Price (yen)')

plt.rcParams['figure.figsize'] = [20, 10]
plt.rcParams['font.size'] = 12

def plt_seasonal_decomposition(data, period):
    res = sm.tsa.seasonal_decompose(data, period = period)
    fig = res.plot()
    plt.show()

plt_seasonal_decomposition(data, period = 100)
