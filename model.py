import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd
import pickle

df = pd.read_csv('hiring.csv')

df['experience'].fillna(0, inplace=True)
df['test_score'].fillna(df['test_score'].mean(), inplace=True)

train_x = df.iloc[:, :3]
print(train_x)

def categorical_to_int(word):
    word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8, 'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, '0':0}
    return word_dict[word]

train_x['experience'] = train_x['experience'].apply(lambda x : categorical_to_int(x))

train_y = df.iloc[:, 4]
print(train_x)
print(train_y)

regressor = LinearRegression()

regressor.fit(train_x, train_y)

pickle.dump(regressor, open('model.pkl', 'wb'))

model = pickle.load(open('model.pkl', 'rb'))
print(model.predict([[2,9,6]]))