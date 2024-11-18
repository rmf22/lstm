
import pandas as pd
from statsmodels.tsa import stattools
import matplotlib.pyplot as plt

from dataset import stock

data = stock['Close']

# Augmented Dickey-Fuller test
def ADF_test(data):
    ADF_data = pd.DataFrame([stattools.adfuller(data)[1]], columns = ['P value'])
    ADF_data['P value'] = ADF_data['P value'].round(decimals = 3) #.astype(str)
    print(ADF_data)

ADF_test(data)
