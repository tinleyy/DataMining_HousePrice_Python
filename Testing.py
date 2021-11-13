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
def CHECK_IS_NUMBER(df):
    return df.select_dtypes(include=['float64', 'int64'])

def CHECK_NOT_NUMBER(df):
    return df.select_dtypes(exclude=['float64', 'int64'])

# check train and test data type is the same

# change the words to int
# df_data['Sex_Code'] = df_data['Sex'].map({'female' : 1, 'male' : 0}).astype('int')

# ------Graph------
# boxplot
# pivot_table
# distplot
# probplot
def salesPriceEffectGraph(df):
    for i in df:
        #sb.boxplot(x='', y="SalePrice", hue="Street", data=data)
        #plt.show()
        return ""
        
def distributionGraph(df):
    return ""

# Show the graph -> lower than 10% -> drop
def MissingRateGraph():
    return ""

# random forest


# Step 1: Read the file from local
train = pd.read_csv(r'C:/Users/ACER/Desktop/data/train.csv', index_col=0)
test = pd.read_csv(r'C:/Users/ACER/Desktop/data/test.csv', index_col=0)

# Step 2: Try to print the data
print("---Head----")
print(train.head())
print(test.head())

# Step 3: Check the size of the data frame
print("---Shape----")
print(train.shape)
print(test.shape)

# Step 4: Check the Dtype of the data frame
print("---DataFrame----")
print(train.info()) 

num = CHECK_IS_NUMBER(train)
not_num = CHECK_NOT_NUMBER(train)
print("Number Column:", len(num.keys()))
print("Column NOT number:", len(not_num.keys()))
for i in not_num.keys(): print(i, end=',', flush=True)

# Check distribution
print(num.describe())

# Check categories
#print(NUMBEROFVALUES(test,'Street'))

# Check the relation
data = train[['MSSubClass', 'SalePrice']].groupby(['MSSubClass'], as_index=False).mean().sort_values(by='SalePrice', ascending=False)
print(data)

# Modeling

# Replace the NA column
train = train.fillna(0)
train = train.replace('NaN',0)
test = test.fillna(0)
test = test.replace('NaN',0)

#train_copy = train.iloc[: , 1:]
#test_copy = test.iloc[: , 1:]

# Combine the data
#data = pd.concat([train['SalePrice'], train['Street']], axis=1, keys=['SalePrice', 'Street'])
all_data = pd.concat([train, test], axis=1)
all_data = all_data.fillna(0)

print(all_data.shape)

# Step x: Change non-num column to number column
print("----ExChange to NUMBER----")
all_data = pd.get_dummies(all_data)
print(all_data)
train = all_data[:len(train)]
test = all_data[len(train):]

train_copy = train.copy()
train_copy = DROP_COLUMN(train_copy, "SalePrice")
test_copy = test.copy()
test_copy = DROP_COLUMN(test_copy, "SalePrice")

# Predict data
reg = LinearRegression().fit(train_copy, train["SalePrice"])
reg.coef_
reg.intercept_
predicted = reg.predict(test_copy)
print(predicted)


index = test.index.tolist()

# Save to csv
preds = pd.DataFrame({'Id' : index, 'SalePrice': predicted})
preds.to_csv('sumbmissionHP.csv', index = False)
