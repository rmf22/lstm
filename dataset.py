import pandas as pd
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))    # 当ファイルが存在するディレクトリに移動

stock = pd.read_csv('stock_price.csv')
stock = stock.rename(columns = {
    '日付け': 'Date',
    '終値': 'Close',
    '始値': 'Open',
    '高値': 'High',
    '安値': 'Low',
    '出来高': 'Turnover',
    '変化率 %': 'Rate of change %'
})                                  # 列名を英語に変更（pyplotでプロット時の表示エラー回避のため）

for index, row in stock.iterrows():
    # 「日付け」列をpandas._libs_tslibs.timestamps.Timestamp型に変換
    stock.loc[index, 'Date'] = pd.to_datetime(row['Date'])#.replace('-', '')

    # 「出来高」列から「M」(百万)と「B」(十億)を除去
    if row['Turnover'][-1] == 'B':    # billion
        stock.loc[index, 'Turnover'] = float(row['Turnover'][:-1]) * 1000
    else:                             # million
        stock.loc[index, 'Turnover'] = float(row['Turnover'][:-1])

    # 「変化率 %」列から「%」を除去
    stock.loc[index, 'Rate of change %'] = float(row['Rate of change %'][:-1])


stock.set_index('Date', inplace = True) # 「日付け」列をインデックスに指定
stock = stock.iloc[::-1]                # 日付けを昇順に変更
