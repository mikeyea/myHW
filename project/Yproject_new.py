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
#Fill null values with 'NA'; error expected
#public_msgs.isnull()
#public_msgs.fillna(value='NA', inplace=True)

#0.2 Determine the number of many top-level messages
public_msgs.count() #3535, total number of messages
public_msgs[public_msgs['id'] == public_msgs['thread_id']].count() #2107, total number of top-level messages
#the following includes 2nd and n-order replies to the original message
public_msgs.thread_id.value_counts() #2160, includes duplicates (replying to own messages)

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
y_data['num_replies'] = cnt_replies # assumes cnt_replies are
#ordered, so I can just add to the dataframe (feels weird)

'''
2. Add features to the dataframe
'''
#2.1 Dummie Variables
'''All values are 1.  Not sure why notnull is not working.
#2.1.1 Determine whether group_id is N/A or not and store to y_data
public_msgs['in_group'] = public_msgs['group_id'].notnull().astype(int)
df = public_msgs[['id','in_group']]
y_data = pd.merge(y_data, df)
#2.1.2 Determine whether top-level message has attachments
public_msgs['has_attach'] = public_msgs['attachments'].notnull().astype(int)
df = public_msgs[['id','has_attach']]
y_data = pd.merge(y_data, df)
'''
#2.1.3 Determine length of body of the message
public_msgs['msg_len'] = public_msgs['body'].str.len()
df = public_msgs[['id','msg_len']]
y_data = pd.merge(y_data, df)
#2.1.4 Determine whether body of the message has 'http' 
public_msgs['has_url'] = public_msgs.body.str.contains('http').astype(int)
df = public_msgs[['id','has_url']]
y_data = pd.merge(y_data, df)
#2.1.5 Natural language processing (txt blob) to detect tone/sentiment of body of the message
#2.1.6 Determine whether body of the message has a question mark 
public_msgs['has_qm'] = public_msgs.body.str.contains("'?'").astype(int)
df = public_msgs[['id','has_qm']]
y_data = pd.merge(y_data,df)        
# visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.pairplot(y_data, x_vars=['msg_len','has_url', 'has_qm'], y_vars='num_replies', size=6, aspect=0.7, kind='reg')

'''
3. OLM Regression
'''
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics

#3.1 create X and y
feature_cols = ['msg_len','has_url','has_qm']
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
metrics.r2_score(y, y_pred)

#3.5 RMSE
def train_test_rmse(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
    y_pred = linreg.predict(X_test)
    return np.sqrt(metrics.mean_squared_error(y_test, y_pred))

#Original Model
train_test_rmse(X, y) #1.239

#3.6 msg_len and has_url
feature_cols = ['msg_len', 'has_url']
X = y_data[feature_cols]
train_test_rmse(X, y) #1.246

#3.7 msg_len and has_qm
feature_cols = ['msg_len', 'has_qm']
X = y_data[feature_cols]
train_test_rmse(X, y) #1.249

#3.8 has_url and has_qm
feature_cols = ['has_url', 'has_qm']
X = y_data[feature_cols]
train_test_rmse(X, y) #1.239

