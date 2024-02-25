''' Case 2'''
from pyspark import SparkContext
#from pyspark.sql import SparkSession
import json
import sys
from itertools import combinations
import time
import random
import math

#sc.stop()

folder = sys.argv[1]
test_filepath = sys.argv[2]
output_filepath = sys.argv[3]

'''folder = '/content/drive/MyDrive/Colab Notebooks/553_hw3_data/'
test_file = 'yelp_val_in.csv'
output_filepath = '/content/outt.csv'
'''

start = time.time()
sc= SparkContext(appName='Case2')
# train data
lines = sc.textFile(folder+'yelp_train.csv') # load train file into rdd
# Skip the header
header = lines.first()
lines = lines.filter(lambda x: x!= header)
# change each line to list
train_data = lines.map(lambda x: x.strip().split(",")).cache()
#train_d = train_data

# test data
lines_ = sc.textFile(folder+test_file) # load train file into rdd
# Skip the header
header = lines_.first()
lines_ = lines_.filter(lambda x: x!= header)
# change each line to list
test_data = lines_.map(lambda x: x.strip().split(",")).cache()

# get features
# business features: bus_id, stars, review_count
bus = sc.textFile(folder+'business.json')
bus_RDD= bus.map(lambda x: json.loads(x)) # load json file to RDD
bus_f = bus_RDD.map(lambda x: (x['business_id'],(x['stars'],x['review_count']))).collectAsMap()
# user features: 
user = sc.textFile(folder+'user.json')
user_RDD= user.map(lambda x: json.loads(x)) # load json file to RDD
user_f = user_RDD.map(lambda x: (x['user_id'],(x['review_count'],x['average_stars']))).collectAsMap()

x_train = []
y_train = []

# train features and rating
for u,b,r in train_data.collect():
  # business in train also in bus feature data
  if b in bus_f.keys():
    b_stars = bus_f[b][0]
    b_r_cnt = bus_f[b][1]
  # bus in train not in feature data, assign 0
  else:
    b_stars = 0
    b_r_count = 0
  # user in train also in user feature data
  if u in user_f.keys():
    u_r_cnt = user_f[u][0]
    u_stars = user_f[u][1]
  # user not in feature data, assign 0
  else:
    u_r_cnt = 0
    u_stars = 0
  # construct the feature for model
  x_train.append([u_r_cnt, u_stars, b_stars, b_r_cnt])
  # construct the target
  y_train.append(r)

# test features
x_test = []
for u,b in test_data.collect():
  # business in train also in bus feature data
  if b in bus_f.keys():
    b_stars = bus_f[b][0]
    b_r_cnt = bus_f[b][1]
  # bus in train not in feature data, assign 0
  else:
    b_stars = 0
    b_r_count = 0
  # user in train also in user feature data
  if u in user_f.keys():
    u_r_cnt = user_f[u][0]
    u_stars = user_f[u][1]
  # user not in feature data, assign 0
  else:
    u_r_cnt = 0
    u_stars = 0
  # construct the feature for model
  x_test.append([u_r_cnt, u_stars, b_stars, b_r_cnt])

import numpy as np
import xgboost as xgb

train_features = np.array(x_train,dtype='float32')
train_ratings = np.array(y_train,dtype='float32')
test_features = np.array(x_test,dtype='float32')
# define XGBoost model
xg_reg = xgb.XGBRegressor(verbosity=0, n_estimators=50, random_state=20, max_depth=5)

# train the model
xg_reg.fit(train_features, train_ratings)

# make predictions
predictions = xg_reg.predict(test_features)

# combine predictions with user and business id
results = combined_data = list(zip(test_data.collect(), predictions))
sc.stop()
end = time.time()
print('Durations:',end-start)
with open(output_filepath, 'w') as file:
  file.write('user_id, business_id, prediction\n')
  for i in results:
    line = '{},{},{}\n'.format(i[0][0], i[0][1], i[1])
    file.write(line)