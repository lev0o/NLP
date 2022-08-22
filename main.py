import pandas as pd
from sklearn.neural_network import N

df = pd.read_csv('Tweets.csv', usecols=['text', 'selected_text', 'sentiment'])
pd.set_option('display.max_columns', None)

for col in df.columns:
    df[col] = df[col].str.lower().str.strip().str.split(' ')

