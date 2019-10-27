import scipy.sparse
import gzip
import numpy as np
import math
import realworlddatahelper as rd
f = gzip.open('docword.enron.txt.gz', 'rb')

# get initial variables
file_content = f.readline()
D = int(file_content)
print(D)
file_content = f.readline()
W = int(file_content)
print(W)
file_content = f.readline()
D_considered = D - 1

# get global counts to be used for tfidf preprocessing
def get_global_counts(f, D_considered):
  f.seek(0)
  file_content = f.readline()
  file_content = f.readline()
  file_content = f.readline()
  doc_id = 1
  file_content = f.readline()
  global_count = np.zeros(W)
  line_nums = file_content.split()
  for i in range(1, D_considered):
    while doc_id == i:
      index = int(line_nums[1]) - 1
      global_count[index] += 1
      file_content = f.readline()
      line_nums = file_content.split()
      doc_id = int(line_nums[0])
  return global_count

# This builds a matrix with all of the original vectors, preprocessed with tfidf.
def build_csr_data(f, D_considered, W, tdidf_used, global_count): 
  doc_id = 1
  f.seek(0)
  file_content = f.readline()
  file_content = f.readline()
  file_content = f.readline()
  passing = 0

  file_content = f.readline()
  line_nums = file_content.split()
  column_indices = []
  row_indices = []
  values = []
  for i in range(1, D_considered):
    linfinity = 0
    if i % 100 == 0:
      print(i)
    linfinity_index = 0
    num = 0
    columns = []
    values_doc = []
    while doc_id == i:
      num += 1
      index = int(line_nums[1]) - 1
      count = int(line_nums[2])
      if tdidf_used:
        count *= math.log(D_considered / global_count[index])
      values.append(count)
      columns.append(index)
      file_content = f.readline()
      line_nums = file_content.split()
      doc_id = int(line_nums[0])
    if num != 0:
      column_indices = column_indices + columns
      rows = [i - 1] * num
      row_indices = row_indices + rows
      values = values + values_doc
  return scipy.sparse.csr_matrix((values, (row_indices, column_indices)), shape=(D_considered, W))

matrix_tdidf = build_csr_data(f, D_considered, W, True, get_global_counts(f, D_considered))
eps = 0.07
m = 750
T = 100
print(rd.get_empirical_values(m, 1, W, matrix_tdidf, T, eps))
