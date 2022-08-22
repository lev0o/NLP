import pandas as pd
import sklearn as sk
import seaborn

df = pd.read_csv('Tweets.csv', usecols=['text', 'selected_text', 'sentiment'])
pd.set_option('display.max_columns', None)

for col in df.columns:
    df[col] = df[col].str.lower().str.strip().str.split(' ')

print(df.head(20))