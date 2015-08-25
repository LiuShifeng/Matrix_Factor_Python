#!/usr/bin/python
#
# Created by Albert Au Yeung (2010)
#
# An implementation of matrix factorization
#
try:
    import numpy
    import math
except:
    print "This implementation requires the numpy module."
    exit(0)

###############################################################################

"""
@INPUT:
    R     : a matrix to be factorized, dimension N x M
    P     : an initial matrix of dimension N x K
    Q     : an initial matrix of dimension M x K
    K     : the number of latent features
    steps : the maximum number of steps to perform the optimisation
    alpha : the learning rate
    beta  : the regularization parameter
    Wm    : the training weight for unobserved rates when used in topK recommendation
@OUTPUT:
    the final matrices P and Q
"""
def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02, Wm = 0.0):
    Q = Q.T
    for step in xrange(steps):
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                eij = R[i][j] - numpy.dot(P[i,:],Q[:,j])
                if float(R[i][j]) == 0.0:
                    for k in xrange(K):
                        eij = Wm * eij
                        oldPik = P[i][k]
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * oldPik - beta * Q[k][j])
                else:
                    for k in xrange(K):
                        oldPik = P[i][k]
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * oldPik - beta * Q[k][j])
        eR = numpy.dot(P,Q)
        e = 0
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if float(R[i][j]) == 0.0:
                    e += Wm * pow(R[i][j] - numpy.dot(P[i,:],Q[:,j]), 2)
                else:
                    e += pow(R[i][j] - numpy.dot(P[i,:],Q[:,j]), 2)
                    for k in xrange(K):
                        e += (beta/2) * ( pow(P[i][k],2) + pow(Q[k][j],2) )
        if e < 0.001:
            break
    print 'R='
    print R
    print 'P='
    print P
    print 'Q='
    print Q.T    
    return P, Q.T

###############################################################################
def RMSE(R,P,Q):
    rmse = 0.0
    N = len(R)
    M = len(R[0])
    for i in range(N):
        for j in range(M):
            if R[i][j] != 0.0:
                rmse += pow(R[i][j] - numpy.dot(P[i,:],Q[j,:]), 2)
    return math.sqrt(rmse/N)

def topK_Hit_Ratio( R,users, items,K=5,relevent_bench=5):
        Hk = 0.0
        recall = 0.0
        Nu = []
        Nku = []
        N = len(R)
        M = len(items)
        sumNku = 0.0
        sumNu = 0.0

        for i in range(N):
            u = []
            uNu = 0
            uNku = 0
            for j in range(M):
                print users[i,:].shape, items[j,:].shape
                u.append(numpy.dot(users[i,:],items[j,:]))
            u.sort(reverse = True)
            for j in range(M):
                if R[i][j] >= relevent_bench:
                    uNu += 1
                    if u.index(numpy.dot(users[i,:],items[j,:]))<K:
                        uNku += 1
            Nu.append(uNu)
            Nku.append(uNku)
        T = 0
        for i in range(N):
            if float(Nu[i]) > 0.0:
                T += 1
                sumNku += float(Nku[i])
                sumNu += float(Nu[i])
                Hk += Nku[i]/Nu[i]
        Hk = Hk/T
        recall = sumNku/sumNu
        return Hk,recall

if __name__ == "__main__":
    R = [
         [5,3,0,1],
         [4,0,0,1],
         [1,1,0,5],
         [1,0,0,4],
         [0,1,5,4],
        ]

    R = numpy.array(R)

    N = len(R)
    M = len(R[0])
    K = 3

    P = numpy.random.rand(N,K)
    Q = numpy.random.rand(M,K)

    nP, nQ = matrix_factorization(R, P, Q, K)

    rmse = RMSE(R,nP,nQ)
    print "rmse ",rmse

    Hk,recall = topK_Hit_Ratio(R,nP,nQ,2,4)
    print Hk,recall
