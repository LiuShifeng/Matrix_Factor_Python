__author__ = 'LiuShifeng'
#!/usr/bin/env python
import sys

from subprocess import *
import resource
import time
import numpy as np
from numpy import linalg
from scipy import sparse
import csv
import math

def LoadMatrix(csvfile):
	data = np.genfromtxt(csvfile, delimiter=',')
	return np.matrix(data)

def LoadSparseMatrix(csvfile):
        val = []
        row = []
        col = []
        select = []
        f = open(csvfile)
        reader = csv.reader(f)
        maxU = 0
        maxI = 0
        for line in reader:
            if int(line[0])>maxU:
                maxU = int(line[0])
            if int(line[1])>maxI:
                maxI = int(line[1])
            row.append( int(line[0])-1 )
            col.append( int(line[1])-1 )
            val.append( float(line[2]) )
            select.append( (int(line[0])-1, int(line[1])-1) )
        return sparse.csr_matrix( (val, (row, col)),shape=(maxU,maxI) ), select

def CalculateError(V, W, H, select):

        diff = V-W*H.T
        error = 0
        for row, col in select:
                error += diff[row, col]*diff[row, col]
        return math.sqrt(error/len(select))

def topK_Hit_Ratio(R ,users ,items ,K = 5 ,relevent_bench = 5):
        Hk = 0.0
        recall = 0.0
        Nu = []
        Nku = []
        N = len(users)
        M = len(items)
        print N,M
        sumNku = 0.0
        sumNu = 0.0

        for i in range(N):
            u = []
            uNu = 0
            uNku = 0
            for j in range(M):
                u.append(np.dot(users[i,:],items[j,:].T))
            u.sort(reverse = True)
            for j in range(M):
                print i,j
                if float(R[i,j]) >= float(relevent_bench):
                    uNu += 1
                    if u.index(np.dot(users[i,:],items[j,:].T))<K:
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

print "---------------------------------------------------"
print "Validation ..."
print "---------------------------------------------------"

W = LoadMatrix(sys.argv[2])
H = LoadMatrix(sys.argv[3])
V, select = LoadSparseMatrix(sys.argv[1])
Hk,recall = topK_Hit_Ratio(V,W,H,K=4)
print "Hk ", Hk, " recall ", recall


