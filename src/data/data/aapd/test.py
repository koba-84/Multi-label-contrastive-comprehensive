import pandas as pd

train = pd.read_csv('train.csv')
dev = pd.read_csv('dev.csv')
test = pd.read_csv('test.csv')
print(train.shape, dev.shape, test.shape)