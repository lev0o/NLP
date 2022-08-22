import pandas as pd

df = pd.read_csv('data/Tweets.csv', usecols=['text', 'selected_text', 'sentiment'])
pd.set_option('display.max_columns', None)

for col in df.columns:
    df[col] = df[col].str.lower().str.strip().str.split(' ')

