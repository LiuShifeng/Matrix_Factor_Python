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

print "---------------------------------------------------"
print "Validation ..."
print "---------------------------------------------------"

W = LoadMatrix(sys.argv[2])
H = LoadMatrix(sys.argv[3])
V, select = LoadSparseMatrix(sys.argv[1])
error = CalculateError(V,W,H,select)
print "Reconstruction RMSE:", error

