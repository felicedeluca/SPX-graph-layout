import networkx as nx
import numpy as np
from numpy import inf
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from LP_with_input import *
from edge_crossing import *
from networkx.drawing.nx_pydot import write_dot
# from plot_layout_statistics import parse_dot_file
from input_functions import *
import math
import sys
import time


def convertEdgeList(G):
    '''
    Converts the edge list
    '''
    edge_list = []
    # Compute edge list
    for e in G.edges():
        u, v = e
        tmp = []
        tmp.append(u)
        tmp.append(v)
        edge_list.append(tmp)
    return edge_list


def getEdgePairAsMatrix(edge_list, X,i,j):
    '''
    This function extracts the edge pair in the form of matrices
    Returns two matrices A and B
    A contains [a1x, a1y; a2x a2y]
    B contains [b1x, b1y; b2x b2y]
    '''
    A = np.zeros((2,2))
    B = np.zeros((2,2))

    i1, i2 = edge_list[i][0], edge_list[i][1]
    j1, j2 = edge_list[j][0], edge_list[j][1]

    A[0,:] = X[i1, :]
    A[1,:] = X[i2, :]

    B[0,:] = X[j1, :]
    B[1,:] = X[j2, :]

    return A,B
