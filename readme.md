### Overview
This is the code repository for the experiments in the following paper: "Understanding Sparse JL for Feature Hashing", to appear at NeurIPS 2019. The paper is available at https://papers.nips.cc/paper/9656-understanding-sparse-jl-for-feature-hashing. 


### Paper Abstract 
Feature hashing and other random projection schemes are commonly used to reduce the dimensionality of feature vectors. The goal is to efficiently project a high-dimensional feature vector living in $\mathbb{R}^n$ into a much lower-dimensional space $\mathbb{R}^m$, while approximately preserving Euclidean norm. These schemes can be constructed using sparse random projections, for example using a sparse Johnson-Lindenstrauss (JL) transform. A line of work introduced by Weinberger et. al (ICML '09) analyzes the accuracy of sparse JL with sparsity 1 on feature vectors with small $\ell_\infty$-to-$\ell_2$ norm ratio. Recently, Freksen, Kamma, and Larsen (NeurIPS '18) closed this line of work by proving a tight tradeoff between $\ell_\infty$-to-$\ell_2$ norm ratio and accuracy for sparse JL with sparsity $1$. 

In this paper, we demonstrate the benefits of using sparsity $s$ greater than $1$ in sparse JL on feature vectors. Our main result is a tight tradeoff between $\ell_\infty$-to-$\ell_2$ norm ratio and accuracy for a general sparsity $s$, that significantly generalizes the result of Freksen et. al. Our result theoretically demonstrates that sparse JL with $s > 1$ can have significantly better norm-preservation properties on feature vectors than sparse JL with $s = 1$; we also empirically demonstrate this finding.


### Experiments 
These experiments evaluate the performance of sparse Johnson-Lindenstrauss transforms on synthetic and real-world datasets. The experiments mainly serve to validate and illustrate the theoretical findings in the paper, and graphs can be found in the paper. Moreover, the rationale behind the experimental design is discussed in the paper. 

To run these experiments, use Python 3. 
### Synthetic Data Experiments
The synthetic data experiments illustrate the tradeoff between the projected dimension $m$, the sparsity $s$, and the bound on vector $\ell_\infty$-to-$\ell_2$ norm ratio needed to achieve a given accuracy and success probability. See synthetic-data.py for the code. 
### Real-World-Data Experiments
These experiments illustrate the tradeoff between dimension $m$, sparsity $s$, accuracy, and failure probability on real-world datasets. These experiments demonstrate the empirical benefits of using $s > 1$ over the standard setting of $s = 1$, which aligns with the theoretical results. 

See news20.py for the code for the News20 dataset. For the Enron dataset, download docword.enron.txt.gz on https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/, save the file as docword.enron.txt.gz, and then see enron.py. 
