import realworlddatahelper as rd

# This is the News20 dataset
from sklearn.datasets import fetch_20newsgroups
newsgroups_train = fetch_20newsgroups(remove=('headers', 'footers', 'quotes'))
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
tdidf_vectorizer = TfidfVectorizer()
vectors_tdidf = tdidf_vectorizer.fit_transform(newsgroups_train.data)

eps = 0.1
m = 200
T = 100
n = 101631
print(rd.get_empirical_values(m, 12, n, vectors_tdidf, T, eps))
