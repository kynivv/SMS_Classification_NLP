# Libraries & Frameworks
import os
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import explained_variance_score as evs
from sklearn.model_selection import train_test_split

from zipfile import ZipFile


# Data Extraction
data_path = 'data'

with ZipFile('SMS.zip') as sms:
    sms.extractall(data_path)


# Data Preprocessing
train_df = pd.read_csv(f'{data_path}/SMS_train.csv', encoding='cp1252')
test_df = pd.read_csv(f'{data_path}/SMS_test.csv', encoding='cp1252')

df = pd.concat([train_df, test_df])

print(df)

df = df.drop('S. No.', axis= 1)

print(df)

X = df['Message_body']
Y = df['Label']

countvec = CountVectorizer()

X = countvec.fit_transform(X)


# Train Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    test_size= 0.25,
                                                    shuffle= True,
                                                    random_state= 24
                                                    )


# Model Training
m = MultinomialNB()

m.fit(X_train, Y_train)


# Model Testing
print(f'Test Accuracy is :{m.score(X_test, Y_test)}')