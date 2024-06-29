import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data = data = pd.read_csv('IBM.csv')
train_data, test_data = train_test_split(data, test_size=0.2 ,random_state = 10)

a = train_data['Close'].value_counts() / len(train_data)
print(a)

b = test_data['Close'].value_counts() / len(test_data)
print(b)