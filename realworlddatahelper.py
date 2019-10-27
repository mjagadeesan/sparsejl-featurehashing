import numpy as np
import scipy
import math

def build_matrix(m, s, n):
  row_ind = []
  col_ind = []
  data = []
  matrices = []
  block_size = math.floor(m/s)
  # T
  sgns = np.random.randint(2, size = s * n)
  sgns = 2 * sgns - 1
  r = np.random.randint(block_size, size = s * n)
  for i in range(n):
    for block in range(s):
      row_ind.append(r[block + i * s] + block * block_size)
      col_ind.append(i)
      data.append(sgns[block + i * s] / math.sqrt(s))
  csr_projection_matrix = scipy.sparse.csr_matrix((data, (col_ind, row_ind)), shape=(n, m))
  return csr_projection_matrix


def get_projected_norms(m, s, n, vectors):
  csr_projection_matrix = build_matrix(m,s,n)
  projections = scipy.sparse.csc_matrix.dot(vectors, csr_projection_matrix)
  projections = projections.power(2)
  norms = scipy.sparse.csr_matrix.sum(projections, axis=-1)
  norms = np.power(norms, 1./2)
  return norms


def get_original_norms(vectors):
  vectors_squared = vectors.power(2)
  original_norms = scipy.sparse.csr_matrix.sum(vectors_squared, axis=-1)
  original_norms = np.power(original_norms, 1./2)
  return original_norms


def get_ratios(m,s,n,vectors):
  # get rid of zero vectors
  original_norms = get_original_norms(vectors)
  (indices, _) = np.nonzero(original_norms)
  original_norms = original_norms[indices]
  projected_norms = get_projected_norms(m, s, n, vectors)
  projected_norms = projected_norms[indices]

  ratios = np.divide(projected_norms, original_norms)
  return np.array(ratios)

def get_delta(ratios, eps):
  passing = ((1 - eps < ratios) & (ratios < 1 + eps)).sum()
  return 1 - (passing / len(ratios))

# This computes for each of T trials, how many of the vectors were within the
# passing region for norm. The mean and stdev are given. 
def get_empirical_values(m, s, n, vectors, T, eps):
  delta_empirical_values = np.zeros(T)
  for i in range(T):
    ratios = get_ratios(m,s,n,vectors)
    delta_emp = get_delta(ratios, eps)
    delta_empirical_values[i] = delta_emp
  delta_sorted = np.sort(delta_empirical_values)
  delta_mean = np.mean(delta_sorted)
  delta_stdev = np.std(delta_sorted) / math.sqrt(T)
  return (delta_mean, delta_stdev)
