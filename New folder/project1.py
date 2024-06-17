import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
df = pd.read_csv('assets/spam.csv', encoding='latin-1') 
#print(df)
data=df.where((pd.notnull(df)), '')
#print(data.head(10))
#data.info()
#print(data.shape)
data.loc[data['v1'].isin(['spam']), 'v1'] = 0
data.loc[data['v1'].isin(['ham']),'v1',]=1
X=data['v2']
Y=data['v1']
#print(X)
#print(Y)
#data=df.where((pd.notnull(df)),'')
x1,x2,y1,y2=train_test_split(X,Y,test_size=0.2,random_state=3)
print(x1.shape)
print(x2.shape)
print(y1.shape)
print(y2.shape)
feature_extratcion= TfidfVectorizer(min_df=1,stop_words='english',lowercase=True)
x_tf=feature_extratcion.fit_transform(x1)
x_tff=feature_extratcion.transform(x2)
y1=y1.astype('int')
y2=y2.astype('int')
#print(x1)
Model = LogisticRegression()
Model.fit(x_tf,y1)
potd= Model.predict(x_tf)
aotd=accuracy_score(y1,potd)
#print(aotd)
# you can input any value between the brackets
input_mail=["Please call our customer service representative on 0800 169 6031 between 10am-9pm as you have WON a guaranteed å£1000 cash or å£5000 prize!"]
input_data_features=feature_extratcion.transform(input_mail)
prediction=Model.predict(input_data_features)
#print(prediction)
if(prediction==0):
    print('SPAM!')
else:
    print('Normal')
