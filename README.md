# MatrixFactorization_SGD_Spark

## Notes

* This code is based on the version of Anttthea(https://github.com/Anttthea). Thanks a lot!
* This code was developed with python version `2.7.8`
* Data is taken from a subsample of the MovieLens 100K dataset. It is in the triples format:
```
<user_1>,<movie_1>,<rating_11>
...
<user_i>,<movie_j>,<rating_ij>
...
<user_M>,<movie_N>,<rating_ij>
```
Here, user i is an integer ID for the ith user, movie j is an integer ID for the jth movie, and rating ij is the rating given by user i to movie j.
For example, the content of the input file to the script will look like:
```
1,3114,4
1,608,4
1,1246,4
2,1357,5
2,3068,4
2,1537,4
...
6040,562,5
6040,1096,4
6040,1097,4
```
* Script eval_acc.log is intended for algorithm performance evaluation by calculating reconstruction error in matrix factorization


## Pre requisites

As pre requisites you need

* python `2.7`
* pyspark installation version `spark-1.3.0-bin-hadoop2.4`

## Big Matrix Factorization using Spark

* This script implements Large-Scale Matrix Factorization with Distributed Stochastic Gradient Descent (DSGD-MF) in Spark.
* The reference paper sets forth a solution for matrix factorization using minimization of sum of local losses. The solution involves dividing the matrix into strata for each iteration and performing sequential stochastic gradient descent within each stratum in parallel. The two losses considered are the plain non-zero square loss and the non-zero square loss with L2 regularization of parameters W and H.
* DSGD-MF is a fully distributed algorithm i.e. both the data matrix V and factor matrices W and H can be carefully split and distributed to multiple workers for parallel computation without communication costs between the workers. Hence, it is a good match for implementation in a distributed in-memory data processing system like Spark.


## Usage

```
pyspark

>>> spark-submit dsgd_mf.py <num_factors> <num_workers> <num_iterations> <beta_value> <lambda_value> <inputV_filepath> <outputW_filepath> <outputH_filepath>
```

Example:

```
>>> spark-submit dsgd_mf.py 100 10 50 0.8 1.0 test.csv w.csv h.csv
>>> python eval_acc.py input/test.csv output/w.csv output/h.csv 
Reconstruction error: 0.826437682942
```

## Reference

```
Gemulla, Rainer, et al. "Large-scale matrix factorization with distributed stochastic gradient descent." Proceedings of the 17th ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 2011.
```
