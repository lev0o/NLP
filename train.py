import pandas as pd
import re
import pickle
import numpy as np

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# parameters
output_file = "models/model.bin"

# data preperation
df = pd.read_csv('data/Tweets.csv', usecols=['text', 'selected_text', 'sentiment'])
pd.set_option('display.max_columns', None)

df = df.drop(df[df['selected_text'].isna()].index.tolist())
df.reset_index(drop=True, inplace=True)

processed_features = []

for sentence in range(0, len(df)):
    # Remove all the special characters
    processed_feature = re.sub(r'\W', ' ', str(df.loc[sentence, 'selected_text']))

    # remove all single characters
    processed_feature= re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)

    # Remove single characters from the start
    processed_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_feature) 

    # Substituting multiple spaces with single space
    processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)

    # Removing prefixed 'b'
    processed_feature = re.sub(r'^b\s+', '', processed_feature)

    # Converting to Lowercase
    processed_feature = processed_feature.lower().strip()

    processed_features.append(processed_feature)

df['selected_text'] = processed_features

df_fulltrain, df_test = train_test_split(df[['selected_text', 'sentiment']], test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_fulltrain, test_size=0.25, random_state=1)

y_train = df_train.sentiment.values
y_val = df_val.sentiment.values
y_test = df_test.sentiment.values

del df_train['sentiment']
del df_val['sentiment']
del df_test['sentiment']

# training
def train(df_train, y_train):
    tfid = TfidfVectorizer()
    X_train = tfid.fit_transform(df_train['selected_text']).toarray()

    model = RandomForestClassifier(random_state=1)
    model.fit(X_train, y_train)

    return tfid, model

def predict(df, tfid, model):
    X = tfid.transform(df['selected_text']).toarray()
    
    y_pred = model.predict(X)
    
    return y_pred

tfid, model = train(df_train, y_train)
predictions = predict(df_val, tfid, model)

# evaluation 
print(accuracy_score(y_val, predictions))
print(classification_report(y_val, predictions))

# saving the model
with open(output_file, 'wb') as f_out:
    pickle.dump((tfid, model), f_out)