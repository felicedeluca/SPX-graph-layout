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


def stress(X, weights, distances, num_nodes):
    '''
    This function computes the stress of an embedding. It takes as input the coordinates X,
    weights (i.e. d_{ij}^(-2)), ideal distances between the nodes, and the number of nodes
    in the graph
    '''

    s = 0.0
    for i in range(0,num_nodes):
    	for j in range(i+1,num_nodes):
    		norm = X[i,:] - X[j,:]
    		norm = np.sqrt(sum(pow(norm,2)))
    		#if distances[i,j]==math.inf:continue
    		s += weights[i,j] * pow((norm - distances[i,j]), 2)

    return s
