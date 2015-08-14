# Matrix_Factor_Python

## Notes for mf.py
* This code is based on the version of Sapphire1211(https://github.com/Sapphire1211). Thanks a lot!
* This code was developed with python version `2.7.8`
* This code is a example to know how SGD works on the matrix factorization.
* This code hold a example in it. You can run it directly.

## Notes for PMF.py
* This code is based on the version of Sapphire1211(https://github.com/Sapphire1211). Thanks a lot!
* This code was developed with python version `2.7.8`
* This code suits measurements of both RMSE and Top-K.The measurement of Top-K is under development.
* Data is taken from a subsample of the MovieLens 100K data set.More details is under below.

## Notes for dsgd_mf.py && eval_acc.py

* This code is based on the version of Anttthea(https://github.com/Anttthea). Thanks a lot!
* This code was developed with python version `2.7.8`
* This code suits measurements of both RMSE and Top-K.The measurement of Top-K is under development.
* Data is taken from a subsample of the MovieLens 100K data set. It is in the triples format:
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
196,242,3
186,302,3
22,377,1
244,51,2
166,346,1
298,474,4
...
474,836,3
537,445,3
37,385,4
```
276,1090,1
13,225,2
12,203,3
```

## Pre requisites

As pre requisites you need

* python `2.7`
* pyspark installation version `spark-1.4.0-bin-hadoop2.4`

## Big Matrix Factorization using Spark

* This script implements Large-Scale Matrix Factorization with Distributed Stochastic Gradient Descent (DSGD-MF) in Spark.
* The reference paper sets forth a solution for matrix factorization using minimization of sum of local losses. The solution involves dividing the matrix into strata for each iteration and performing sequential stochastic gradient descent within each stratum in parallel. The two losses considered are the plain non-zero square loss and the non-zero square loss with L2 regularization of parameters W and H.
* DSGD-MF is a fully distributed algorithm i.e. both the data matrix V and factor matrices W and H can be carefully split and distributed to multiple workers for parallel computation without communication costs between the workers. Hence, it is a good match for implementation in a distributed in-memory data processing system like Spark.


## Usage

```
>>> spark-submit dsgd_mf.py <num_factors> <num_workers> <num_iterations> <beta_value> <lambda_value> <Wm_value> <inputV_filepath> <outputW_filepath> <outputH_filepath>
```

Example:

```
>>> spark-submit dsgd_mf.py 100 10 50 0.8 1.0 0.2 test.csv w.csv h.csv
>>> python eval_acc.py input/test.csv output/w.csv output/h.csv 
Reconstruction error: 0.826437682942
```

## Reference

```
Gemulla, Rainer, et al. "Large-scale matrix factorization with distributed stochastic gradient descent." Proceedings of the 17th ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 2011.
```
