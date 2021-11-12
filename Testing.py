"""
numpy: Python數學計算基本模組 ，縮寫:np
pandas: Python表格計算基本模組 ，縮寫:pd
matplotlib.pyplot: Python繪製圖片基本模組 ，縮寫:plt
seaborn: Python更漂亮繪製圖片模組 ，縮寫:sb
warnings: Python管理錯誤訊息模組

sklearn.preprocessing:先針對資料做數據清洗
sklearn.future_selection:決定哪些資料是有價值的
sklearn.model_selection:決定我們要使用哪些model分析資料
sklearn.ensemble:決定引入哪些集成分類器，集成使用多種學習算法來獲得比單獨使用任何學習算法更好的預測性能

ipython.display: 讓我們繪製的圖片可以正常顯示
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import warnings

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_selection import RFECV
from sklearn.model_selection import cross_val_score, StratifiedKFold, learning_curve, train_test_split

from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV, ElasticNet, ElasticNetCV
# import lib - Numpy, Scipy, Sympy, Pandas

# check NULL or 0 or NA or "" value
def NULL_VALUES(df):
    return df.isnull().sum()

# drop the column if the data is useless
def DROP_COLUMN(df, colName):
    df.drop(columns=colName,inplace= True)
    return df

# check the length of dataset 
# len(test)
# len(train)

# check the number of a column
def NUMBEROFVALUES(df, colName):
    return df[colName].value_counts()

# check Data Type

# change the words to int
# df_data['Sex_Code'] = df_data['Sex'].map({'female' : 1, 'male' : 0}).astype('int')

# Show the graph
def salesPriceEffectGraph(df):
    for i in df:
        #sb.boxplot(x='', y="SalePrice", hue="Street", data=data)
        #plt.show()
        return ""
        
# Show the graph
def distributionGraph(df):
    return ""

# Show the graph -> lower than 10% -> drop
def MissingRateGraph():
    return ""

# random forest

# Read the file from local
train = pd.read_csv(r'C:/Users/ACER/Desktop/data/train.csv', index_col=0)
test = pd.read_csv(r'C:/Users/ACER/Desktop/data/test.csv', index_col=0)

# Try to print the data
train.head()
test.head()

train.shape
test.shape

print(train.info())
print(train.describe())

print(NULL_VALUES(test))
print(NUMBEROFVALUES(test,'Street'))

# Combine the data
data = pd.concat([train['SalePrice'], train['Street']], axis=1, keys=['SalePrice', 'Street'])
print(data)

# Check the relation
data = train[['MSSubClass', 'SalePrice']].groupby(['MSSubClass'], as_index=False).mean().sort_values(by='SalePrice', ascending=False)
print(data)

# Modeling


# Predict data
#y_hat_test = reg.predict(test_input)
#predicted = np.exp(y_hat_test)

# Save to csv
#preds = pd.DataFrame({'Id' : test['Id'], 'SalePrice': predicted})
#preds.to_csv('sumbmissionHP.csv', index = False)
