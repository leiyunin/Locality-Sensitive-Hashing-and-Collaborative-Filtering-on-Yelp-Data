''' Case 1'''
from pyspark import SparkContext
#from pyspark.sql import SparkSession
import json
import sys
from itertools import combinations
import time
import random
import math

#sc.stop()

train_filepath = sys.argv[1]
test_filepath = sys.argv[2]
output_filepath = sys.argv[3]

'''train_filepath = '/content/yelp_train.csv'
test_filepath = '/content/yelp_val_in.csv'
output_filepath = '/content/out.csv'
'''
start = time.time()
sc= SparkContext(appName='Case1')
# train data
lines = sc.textFile(train_filepath) # load train file into rdd
# Skip the header
header = lines.first()
lines = lines.filter(lambda x: x!= header)
# change each line to list
train_data = lines.map(lambda x: x.strip().split(",")).cache()

# test data
lines_ = sc.textFile(test_filepath) # load train file into rdd
# Skip the header
header = lines_.first()
lines_ = lines_.filter(lambda x: x!= header)
# change each line to list
test_data = lines_.map(lambda x: x.strip().split(",")).cache()
# get train business id and test business id
train_bus = train_data.map(lambda x: (x[1])).distinct()
test_bus = test_data.map(lambda x: (x[1])).distinct()
all_bus = train_bus.union(test_bus).distinct().zipWithIndex() # union all business id for train and test and assign index
all_bus_dict = all_bus.collectAsMap()

# get train user id and test user id
train_user = train_data.map(lambda x: (x[0])).distinct()
test_user = test_data.map(lambda x: (x[0])).distinct()
all_user = train_user.union(test_user).distinct().zipWithIndex() # union all business id for train and test and assign index
all_user_dict = all_user.collectAsMap()

# prepare train and test rdd for Pearson calculation
# get businese_id and user_id, format {bus_id: (user_id, rating)}
train_bus = train_data.map(lambda x: (all_bus_dict[x[1]],(all_user_dict[x[0]],float(x[2])))).groupByKey().mapValues(list).collectAsMap()
#get train and average ratine for each user
# '3MntE_HWbNNoyiLGxywjYA': 3.4
train_bus_avg = train_data.map(lambda x: (all_bus_dict[x[1]],float(x[2]))).mapValues(lambda x: (x, 1)).reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1])).mapValues(lambda x: x[0] / x[1])
train_bus_avg = train_bus_avg.collectAsMap()
#get a train user id and business id buckets, format {user_id, (business_id, rating)}

train_user = train_data.map(lambda x: (all_user_dict[x[0]],(all_bus_dict[x[1]],float(x[2])))).groupByKey().mapValues(list).collectAsMap()
train_u_avg = train_data.map(lambda x: (all_user_dict[x[0]],float(x[2]))).mapValues(lambda x: (x, 1)).reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1])).mapValues(lambda x: x[0] / x[1])
train_u_avg = train_u_avg.collectAsMap()

# get businese_id and user_id, format (bus_id,user_id)
test = test_data.map(lambda x: (all_bus_dict[x[1]],all_user_dict[x[0]]))#.collect()#.repartition(8)


'''Pearson and Prediction'''
def calculate_predict(b, u):
    # check if both business and user are new
    if b not in train_bus and u not in train_user:
        return (b, u, 3.75)  # Assign a default value
    
    # check if business is new but user is old
    if b not in train_bus and u in train_user:
        return (b, u, train_u_avg.get(u, 3.5))  # Use user's average rating
    
    # check if user is new but business is old
    if u not in train_user:
        return (b, u, train_bus_avg.get(b, 3.5))  # Use business's average rating
    
    # initialize numerator and denominator for Pearson correlation calculation
    pred_nu = 0
    pred_de = 0
    
    # find all business this user has rated and users for the target business
    all_bus_ratings = train_user[u]  # List of (business_id, rating) pairs for the user
    b2_ratings = dict(train_bus[b])  # List of (user_id, rating) pairs for the business
    
    # calculate average ratings for the target business
    b2_avg = train_bus_avg.get(b, 3.5)

    top_similar_businesses = []
    # iterate through each business rated by the user
    for b1, b1_rating in all_bus_ratings:
      
        # check if the user also rated the same business for which we are predicting
      if b1 in train_bus:
          b1_ratings = train_bus[b1]  # list of (user_id, rating) pairs for business b1
            
            # find common users who rated both business b1 and the target business b
          common_users = [user_id for user_id, _ in b1_ratings if user_id in b2_ratings]

          #if len(common_users)==2: # not enough information
            # continue
            #top_similar_businesses.append((b1, w))
            #continue
          #elif len(common_users)==2:

          if len(common_users)>2:
              #print('couser',common_users)
                #print('b1',b1)
                # Calculate Pearson correlation for common users
              w_nu = sum((rating - train_bus_avg.get(b1, 3.5)) * (b2_ratings[user_id] - b2_avg)
                           for user_id, rating in b1_ratings if user_id in common_users)
              w_de_1 = sum((rating - train_bus_avg.get(b1, 3.5)) ** 2
                             for _, rating in b1_ratings if _ in common_users)
              w_de_2 = sum((b2_ratings[user_id] - b2_avg) ** 2
                             for user_id, _ in b1_ratings if user_id in common_users)
                #print(w_nu)
              if w_de_1 != 0 and w_de_2 != 0:
                  w = w_nu / (math.sqrt(w_de_1) * math.sqrt(w_de_2))
                  top_similar_businesses.append((b1, w))
          else: # If no or only one co_user, assign a pearson corelation to it
            w = (5-abs(train_bus_avg.get(b1,3.5)-train_bus_avg.get(b,3.5)))/5
            top_similar_businesses.append((b1, w))
    top_similar_businesses.sort(key=lambda x: x[1], reverse=True)
    top_similar_businesses = top_similar_businesses[:15] # get the top 15 neighbers
    #print(top_similar_businesses)

    
    # Calculate the final prediction using only the top N similar businesses
    for b1, w in top_similar_businesses:
     b1_rating = dict(train_user[u])[b1]
     #print(b1_rating)  # Rating given by user u for business b1
     pred_nu += w * (b1_rating)
     #print('nu',pred_nu)
     pred_de += abs(w)
     #print('de',pred_de)
    
    # Calculate the final prediction
    if pred_de == 0:
     predict = 3.5  # Assign a default value if no correlation found
    else:
     predict = pred_nu / pred_de  
    #print(predict)
    return (b, u, predict)

res=test.map(lambda x: calculate_predict(x[0],x[1])).collect()
sc.stop()
end = time.time()
print('Duration:',end-start)
reverse_user_id_dict = {v: k for k, v in all_user_dict.items()}  # Reversing keys and values
reverse_business_id_dict = {v: k for k, v in all_bus_dict.items()}
with open(output_filepath, 'w') as file:
  file.write('user_id, business_id, prediction\n')
  for i in res:
    line = '{},{},{}\n'.format(reverse_user_id_dict[i[1]], reverse_business_id_dict[i[0]], i[2])
    file.write(line)
