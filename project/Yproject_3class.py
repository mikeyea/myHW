# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 15:41:25 2015

@author: HHS
"""
import pandas as pd
import sys
reload(sys)
sys.setdefaultencoding('utf8')
messages = pd.read_csv('Messages.csv', na_filter=False)
messages.shape
messages.describe()
messages.dtypes
'''
0. Data analysis and remove private messages posted private groups
'''
#0.1 Remove private messages or messages posted in private groups 
messages.in_private_group.sum()
messages.in_private_conversation.sum()
public_msgs = messages[(messages.in_private_group == False) & (messages.in_private_conversation == False)]
public_msgs = public_msgs.drop(['in_private_group', 'in_private_conversation'], axis=1)
public_msgs = public_msgs.replace('', 'NA')
#0.2 Determine the number of many top-level messages
public_msgs.count() #15664, total number of messages
public_msgs[public_msgs['id'] == public_msgs['thread_id']].count() #9616, total number of top-level messages
#the following includes 2nd and n-order replies to the original message
public_msgs.thread_id.value_counts() #9762, includes duplicates (replying to own messages)

'''
1. Create response variable column in a dataframe
'''
#1.1 Initialize a dataframe
y_data = pd.DataFrame()
y_data = y_data.fillna(0)
#1.2 Add number of replies to each original message
#1.2.1 Populate dataframe with top-level message ID
top_msg_id = []
for index, row in public_msgs.iterrows():
    if row.id == row.thread_id:
        top_msg_id.append(row.id)
y_data['id'] = top_msg_id
#1.2.2 Create a function that returns the number of replies to a given top-level message ID
def get_reply_counts(id):    
    public_msgs['num_reply'] = public_msgs.replied_to_id.str.contains(str(id)).astype(int)
    return public_msgs['num_reply'].sum()    
    
#1.2.3 Determine the number of replies to each top-level message and store it to the dataframe
cnt_replies = []
for row in y_data.id:
    cnt_replies.append(get_reply_counts(row))
y_data['num_replies'] = cnt_replies 

'''
2. Add features to the dataframe
'''

#2.1 Determine whether group_id is N/A or not and store to y_data
import numpy as np
public_msgs['in_group'] = np.where(public_msgs['group_id']=='NA',0,1)
df = public_msgs[['id','in_group']]
y_data = pd.merge(y_data, df)
#2.2 Determine whether top-level message has attachments
public_msgs['has_attach'] = np.where(public_msgs['attachments']=='NA',0,1)
df = public_msgs[['id','has_attach']]
y_data = pd.merge(y_data, df)
#2.3 Determine length of body of the message
public_msgs['msg_len'] = public_msgs['body'].apply(len)
df = public_msgs[['id','msg_len']]
y_data = pd.merge(y_data, df)
#2.4 Determine whether body of the message has 'http' 
public_msgs['has_url'] = public_msgs.body.str.contains('http').astype(int)
df = public_msgs[['id','has_url']]
y_data = pd.merge(y_data, df)
#2.5 Natural language processing (txt blob) to detect tone/sentiment of body of the message
from textblob import TextBlob, Word
def detect_sentiment(text):
    return TextBlob(text.decode('utf-8')).sentiment.polarity
public_msgs['sentiment'] = public_msgs.body.apply(detect_sentiment)
df = public_msgs[['id','sentiment']]
y_data = pd.merge(y_data, df)
#2.6 Determine whether body of the message has a question mark 
public_msgs['has_qm'] = public_msgs.body.str.contains("'?'").astype(int)
df = public_msgs[['id','has_qm']]
y_data = pd.merge(y_data, df)
#2.7 Key words ["experience", "opportunity", "interest"] use apply(key_word_search)
import re
def search_key_words(text):
    return len(re.findall(r"(experience|opportunity|interest)", text))
public_msgs['has_key_word'] = public_msgs.body.apply(search_key_words)
df = public_msgs[['id','has_key_word']]
y_data = pd.merge(y_data,df)  
#2.8 Message Author Age
#2.8.1 Import user information and merge with message dataframe
df = public_msgs[['id','sender_id','created_at']]
y_data = pd.merge(y_data,df)
users = pd.read_csv('Users.csv', na_filter=False)
users = users.replace('', 'NA')
users.rename(columns={'id':'sender_id'}, inplace=True)
df = users[['sender_id','job_title','joined_at','state']]
y_data = pd.merge(y_data,df)
#2.8.2 Convert object type to DTG
y_data['joined_at'] = pd.to_datetime(y_data.joined_at)
y_data['created_at'] = pd.to_datetime(y_data.created_at)
#perform calculation and apply to the dataframe
y_data['author_age'] = (y_data.created_at - y_data.joined_at).dt.days
# visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
y_data.num_replies.plot(kind='hist', bins=20, title='Histogram of No. Replies to a Top-Level Message')
plt.xlabel('Number of Replies')
plt.ylabel('Frequency') 
sns.pairplot(y_data, x_vars=['msg_len','has_url', 'has_qm', 'author_age'], y_vars='num_replies', size=6, aspect=0.7, kind='reg')
sns.pairplot(y_data, x_vars=['sentiment','has_key_word','in_group','has_attach'], y_vars='num_replies', size=6, aspect=0.7, kind='reg')
'''
3. OLM Regression
'''
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics

#3.1 create X and y
feature_cols = ['in_group','has_attach','msg_len','has_url','sentiment','has_qm','has_key_word','author_age']
X = y_data[feature_cols]
y = y_data['num_replies']

#3.2 instantiate and fit
linreg = LinearRegression()
linreg.fit(X, y)

#3.3 print the coefficients
linreg.intercept_
linreg.coef_

#3.4 R2 value
y_pred = linreg.predict(X)
metrics.r2_score(y, y_pred) #.05

#3.5 RMSE
def train_test_rmse(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
    y_pred = linreg.predict(X_test)
    return np.sqrt(metrics.mean_squared_error(y_test, y_pred))

#Original Model
train_test_rmse(X, y) #1.514

#3.6 msg_len, has question mark, sentiment, has key words, in a group
feature_cols = ['msg_len', 'has_qm', 'sentiment', 'has_key_word', 'in_group']
X = y_data[feature_cols]
train_test_rmse(X, y) #1.522

#3.7 msg_len and has_qm
feature_cols = ['msg_len', 'has_qm']
X = y_data[feature_cols]
train_test_rmse(X, y) #1.534

#3.8 has_url and has_qm
feature_cols = ['has_url', 'has_qm']
X = y_data[feature_cols]
train_test_rmse(X, y) #1.533

'''
4. Logistic Regression using DTM of body of the message as a feature and other features
'''
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import scipy as sp
#4.1 Convert the response variable to classes
def convert_to_class(num_replies):
    num_replies_class = int    
    if num_replies > 1:
        num_replies_class = 2
    elif num_replies == 1:
        num_replies_class = 1
    else:
        num_replies_class = 0
    return num_replies_class
y_data['num_replies_class'] = y_data.num_replies.apply(convert_to_class)
#Binary: 0 reply to post: 0; 1 or more replies to post: 1
#y_data['num_replies_class'] = np.where(y_data.num_replies>0,1,0)
#4.2 Add the body of the message to the dataframe
df = public_msgs[['id','body']]
y_data = pd.merge(y_data,df)
#4.3 Split the new DataFrame into training and testing sets
feature_cols = ['body', 'job_title', 'in_group', 'has_attach','msg_len', 'has_qm', 'has_key_word', 'author_age']
X = y_data[feature_cols]
y = y_data['num_replies_class']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
#4.4 Use CountVectorizer with body of the message only
vect = CountVectorizer(ngram_range=(1, 2), min_df=2)
train_dtm = vect.fit_transform(X_train[:, 0])
test_dtm = vect.transform(X_test[:, 0])
#4.4b Use CountVectorizer with job_title of the author only
train_dtm_jt = vect.fit_transform(X_train[:, 1])
test_dtm_jt = vect.transform(X_test[:, 1])
#4.5 Cast other feature columns to float and convert to a sparse matrix
extra = sp.sparse.csr_matrix(X_train[:, 2:].astype(float))
#4.6 Combine sparse matrices
train_dtm_extra = sp.sparse.hstack((train_dtm, train_dtm_jt, extra))
#4.7 Repeat for testing set
extra = sp.sparse.csr_matrix(X_test[:, 2:].astype(float))
test_dtm_extra = sp.sparse.hstack((test_dtm, test_dtm_jt, extra))
#4.8 Use logistic regression with the body of the message
logreg = LogisticRegression(C=1e9)
logreg.fit(train_dtm, y_train)
y_pred_class = logreg.predict(test_dtm)
metrics.accuracy_score(y_test, y_pred_class) #.716
#4.8b Use logistic regression with thre job title of the author
logreg = LogisticRegression(C=1e9)
logreg.fit(train_dtm_jt, y_train)
y_pred_class = logreg.predict(test_dtm_jt)
metrics.accuracy_score(y_test, y_pred_class) #.785
#4.9 Use logistic regression with all features
logreg = LogisticRegression(C=1e9)
logreg.fit(train_dtm_extra, y_train)
y_pred_class = logreg.predict(test_dtm_extra)
y_pred_prob = logreg.predict_proba(test_dtm_extra)
metrics.accuracy_score(y_test, y_pred_class) #.770
metrics.confusion_matrix(y_test, y_pred_class)
#4.10 Null Accuracy
1-y_test.mean() #.714
