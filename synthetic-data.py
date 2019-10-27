import matplotlib.pyplot as plt
import scipy.stats
import numpy as np
import math
from random import randint

# This computes the fraction of norms of projected vectors ||Mx||_2^2 
# within 1 +- eps. The vector consists of k 1s and the rest 0s. 
# This compute {hat}(m, eps, delta, s). A new matrix is sampled in each trial. 
# The matrix is sampled by sampling an index for each of k columns corresponding
# to the support of the vector for each block. The values are sampled with 
# np.random.randint(block_size, size=k_val) for each block for each trial
# and np.random.randint(2, size=k_val). 
def get_delta(k_val, s, m_val, S, eps):
  max_norm = (1 + eps) * math.sqrt(k_val)
  min_norm = (1 - eps) * math.sqrt(k_val)
  passing = 0
  block_size = math.floor(m_val/s)
  batch_size = 5
  num_batches = math.ceil(S / batch_size)
  for batch in range(num_batches):
    projection = np.zeros((batch_size, m_val))
    print("batch: " + str(batch))
    for j in range(batch_size):
      # choose 1 entry per block_size block of column
      # handle projection block-by-block
      for b in range(s):
        # generate nonzero entries and signs in block for k relevant columns
        r = np.random.randint(block_size, size=k_val)
        sgns = np.random.randint(2, size=k_val)
        sgns = 2 * sgns - 1
        # iterate through k nonzero entries of vector
        for i in range(k_val):
          # update projection vector
          projection[j][r[i] + b * block_size] += sgns[i]
    # compute the l2 norms for each block and then aggregate over blocks
    norms = np.sum(np.abs(projection)**2,axis=-1)**(1./2)
    norms = norms / math.sqrt(s)
    # compute the percentage of trials that were passing
    passing += ((min_norm < norms) & (norms < max_norm)).sum()
  return 1 - (passing / (batch_size * num_batches))

def get_v(s, m, S, eps, delta, k_vals):
  for i in range(len(k_vals)):
    # compute the delta value on each vector
    delta_val = get_delta(k_vals[i], s, m, S, eps)
    print("kval: " + str(k_vals[i]) + ", delta val: " + str(delta_val))
    print("delta: " + str(delta))
    # stop when delta exceeds threshold
    if delta_val > delta:
      if i == 0:
        return 0
      else: 
        return 1 / math.sqrt(k_vals[i - 1])
  return 1 / math.sqrt(k_vals[len(k_vals) - 1])


# set up k values
k_values = [math.ceil(1.5 ** i) for i in range(0, 4)]
k_values.append(3)
k_values.append(5)
k_values.append(6)
k_values.append(7)
# k_values.append(8)
# k_values.append(9)
k_values = list(set(k_values))
k_values.sort()
k_values.reverse()

T =  10
eps = 0.02
delta = 0.05
print("epsilon: " + str(eps))
print("delta: " + str(delta))
print("T value: " + str(T))

# uncomment the next line to compute hat{v}(m, epsilon, delta, s)
s = 2
m = 18000
v_val = get_v(s, m, T, eps, delta, k_values)
print(v_val)

