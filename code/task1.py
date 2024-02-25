''' version 2'''
from pyspark import SparkContext
#from pyspark.sql import SparkSession
import json
import sys
from itertools import combinations
import time
import random


input_filepath = sys.argv[1]
output_filepath = sys.argv[2]

'''input_filepath = '/content/yelp_train.csv'
output_filepath = '/content/out.csv'
'''

start = time.time()
sc= SparkContext(appName='Task1')
lines = sc.textFile(input_filepath) # load csv file into rdd
# Skip the header
header = lines.first()
lines = lines.filter(lambda x: x!= header)
# change each line to list
lines = lines.map(lambda x: x.strip().split(","))
# get businese_id and user_id, format (bus_id,user_id)
lines = lines.map(lambda x: (x[1],x[0]))

# assign index for distinct user_id
user_id = lines.map(lambda x:x[1]).distinct().zipWithIndex()
# assign index for distinct business_id
# didn't use in the following codes, keep for get the num of business
bus_id = lines.map(lambda x:x[0]).distinct().zipWithIndex()

# save the id and index into dict
user_dict = user_id.collectAsMap()
#bus_dict = bus_id.collectAsMap()

# use these two index dict to map the real data & get rated user_id for each businese_id
index_data = lines.map(lambda x: (x[0],user_dict[x[1]])).groupByKey().mapValues(lambda x: set(x))
index_dict = index_data.collectAsMap()

# get num of user and business for hashing
num_user = user_id.count()
num_bus = bus_id.count()

# hash function
def hash_functions(a, b, m):
    def hash_matrix(x):
        return (a * x + b) % m
    return hash_matrix
ab_values = []
hash_num = 50 # 50 hash funcs total
for i in range(hash_num):
  a = random.randint(1, 100)
  b = random.randint(1, 100)
  ab_values.append((a, b))

# Minhash

  # get signture matrix
def sig(d):
  #print('ab',ab_values)
  hash_func = [hash_functions(a, b, num_user) for a, b in ab_values]
  hash_min = []
  for hashs in hash_func: # for each hash function
    idx=[]
    for j in d: # get each user index
      idx.append(hashs(j))
    #print(idx)
    min_idx = min(idx)
    hash_min.append(min_idx)
  return hash_min

sig_M = index_data.flatMap(lambda x: [(x[0], sig(x[1]))]).cache()

# LSH

# b * r = 50
b = 25  # Number of bands
r = 2   # Number of rows in each band

def divide_into_bands(sig_m, b, r):
    divided_bands = []
    for business_id, hash_values in sig_m:
        # Divide hash values into bands & hash the bands into num_bus buckets
        bands = [hash(tuple(hash_values[i:i + r])) % num_bus for i in range(0, len(hash_values), r)]
        # Generate tuples with band index as the key and business ID along with hash values
        for i, band in enumerate(bands):
            divided_bands.append(((i, band),business_id))
    return divided_bands

# Use flatMap to divide hash values into bands, format (business_id, list_of_hash_values)
bands = sig_M.flatMap(lambda x: divide_into_bands([x], b, r))

def generate_combinations(pair):
    key, values = pair
    # Generate combinations of length 2 for the values
    value_combinations = list(combinations(values, 2))
    # Return the key along with the combinations
    return value_combinations
candi_wl = bands.groupByKey().mapValues(list).filter(lambda x: len(x[1])>1).flatMap(generate_combinations).distinct().collect()

def calculate_jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    similarity = intersection / union if union != 0 else 0
    return similarity

sc.stop()

candi=[]
for i in candi_wl:
  c=calculate_jaccard_similarity(index_dict[i[0]],index_dict[i[1]])
  if c>=0.5:
    candi.append((sorted(i),c))
end = time.time()
print('Duration: ',end-start)
with open(output_filepath, 'w') as file:
  file.write('business_id_1, business_id_2, similarity\n')
  for i in sorted(candi):
    line = '{},{},{}\n'.format(i[0][0], i[0][1], i[1])
    file.write(line)
