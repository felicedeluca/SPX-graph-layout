import networkx as nx
import numpy as np
from numpy import inf
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from LP_with_input import *
from edge_crossing import *
from networkx.drawing.nx_pydot import write_dot
from plot_layout_statistics import parse_dot_file
from input_functions import *
import math
import sys
import time

#if len(sys.argv)<4:
#	print('usage:python3 main_reyan.py normalize(0/1) file_prefix(er_10_0.6) output_folder')
#	quit()

EPSILON = 0.000001
# K = 100000

#NUM_ITERATIONS = 2
#NUM_ITERATIONS = 3
# NUM_ITERATIONS = 10
# NUM_ITERATIONS = 5
NUM_ITERATIONS = 0

USE_NUM_ITERS = True
IS_LOG_COST_FUNCTION = True
#NORMALIZE = int(sys.argv[1])
NORMALIZE = 0

# K = 100000
K = 1
#K = 1000
# W = 0
W = 1

# NUM_RUNS = 5# N
# NUM_RUNS = 2
# NUM_RUNS = 10
# NUM_RUNS = 6
#NUM_RUNS = 1


# G = []#graph
# n = []
# m = []
edge_list = []
distances = []
X_curr = []
X_prev = []
weights = []
penalties = []
u_params = []
gammas = []

USE_INITIAL = ''

# 5 runs for each parameter
# 10 different parameters combination of K and W
# W = [0, 1]
# K = [1, 10, 100, 1000, 10000]
# Cost Function for every run
# Number of Crossings

#FILENAME = 'er_10_0.6'
#FILENAME = sys.argv[2]
#OUTPUT_FOLDER = sys.argv[3]
#FILENAME = "er_50_0.1"
#OUTPUT_FOLDER = "profile_smart_algorithm"
#FILENAME = "input1"
#OUTPUT_FOLDER = "input_angle"
OUTPUT_FOLDER = ""
FILENAME = ""

#G = nx.erdos_renyi_graph(10, 0.4)
#G = build_networkx_graph(OUTPUT_FOLDER+'/'+FILENAME+'.txt')
#n = G.number_of_nodes()
#m = G.number_of_edges()
n = 0
m = 0

# W_start = 0
#W_start = 1
#W_end = 2

#K_start = -5
#K_end = 6

#K_start = 0
#K_end = 1

#NUMBER_OF_CROSSINGS = -np.ones((W_end-W_start, K_end-K_start, NUM_RUNS));
#COST_FUNCTIONS = -np.ones((W_end-W_start, K_end-K_start, NUM_RUNS, NUM_ITERATIONS));
#X_new = np.zeros((W_end-W_start, K_end-K_start, NUM_RUNS, n, 2));
#init_X = np.zeros((W_end-W_start, K_end-K_start, NUM_RUNS, n, 2));

NUMBER_OF_CROSSINGS = []
COST_FUNCTIONS = []
X_new = []
init_X = []

total_u_gamma_time = 0
total_gradient_descent_time = 0
#total_stress_time = 0
#cdef int total_sum_penalty_time = 0
#total_modified_cost_time = 0
total_penalties_after_grad_desc_time = 0
#cdef int total_getEdgePairAsMatrix_time = 0
#cdef int total_getIntersection_time = 0
#cdef int total_getAngleLineSeg_time = 0
#cdef int total_just_sumPenalty_time = 0
#cdef int total_doIntersect_time = 0

THIS_GD_OPTION = 'VANILLA'
THIS_ALPHA = 1e-3
THIS_NUM_ITERS = 100


def runOptimizer(G, W, K, wi, ki):
	###### Initialize and load the graphs; Compute the weights and distances

	# G = nx.petersen_graph()
	# G = nx.complete_graph(10)
	# G = nx.complete_graph(7)
	# G = nx.wheel_graph(9)

	global edge_list
	global distances
	global X_curr
	global X_prev
	global weights
	global penalties
	global u_params
	global gammas
	global n
	global m

	n = G.number_of_nodes()
	m = G.number_of_edges()
	#edge_list = G.edges()
	for e in G.edges():
		u, v = e
		tmp = []
		tmp.append(u)
		tmp.append(v)
		edge_list.append(tmp)

	distances = nx.floyd_warshall(G)

	# Initialize the coordinates randomly in the range [-50, 50]
	X_curr = np.random.rand(n,2)*100 - 50

	if USE_INITIAL != '0':
		#pos = nx.nx_agraph.graphviz_layout(G)
		if USE_INITIAL[len(USE_INITIAL)-4:len(USE_INITIAL)]=='.txt':
			dummy, node_coords, edge_list = take_input(OUTPUT_FOLDER+'/'+USE_INITIAL)
		else:
			node_coords, edge_list = parse_dot_file(OUTPUT_FOLDER+'/'+USE_INITIAL)
		# Copy the coordinates from pos to X_curr
		for i in range(0,n):
			#X_curr[i] = pos[i]
			tmp = np.zeros((2))
			tmp[0] = node_coords[i][0]
			tmp[1] = node_coords[i][1]
			X_curr[i] = tmp

	init_X[wi][ki] = X_curr

	# plotGraphandStats(X_curr)

	# Z=np.copy(X_curr)
	X_prev = np.copy(X_curr)

	# Copy the distances into a 2D numpy array
	distances = np.array([[distances[i][j] for j in distances[i]] for i in distances])
	# weights = 1/(d^2)
	weights = 1/pow(distances,2)
	weights[weights == inf] = 0


	# Define: penalties, u_params, gammas, edgesID


	# penalties: a 2D array containing the penalties for each possible edge pair
	# For now the penalties start with 0 and gradually increase by 1 in the next iteration
	# if the crossing persists.

	penalties = np.zeros((m, m))

	# u_params: a 3D array containing the u vectors for each edge pair
	u_params = np.zeros((m, m, 2))

	# gammas: a 2D array containing the gamma values for each possible edge pair
	gammas = np.zeros((m, m))

	# all these variables need to be accessed as a 2D array
	# with the edge pair as the i,j index of the 2D array.

	#The 2D array is of size M*M where M is the number of edges. Max edges = n*(n-1)/2
	# for complete graph

	return optimize(X_curr, wi, ki, W, K, G)
	# plotGraphandStats(X_curr)


#This function returns the nodes of an edge given its index in the edge list
def getNodesforEdge(index):
	#print(index)
	#print(edge_list)
	#print('type(edge_list)')
	#print(type(edge_list))
	#print('type(edge_list[index])')
	#print(type(edge_list[index]))
	#print(str(edge_list[index]))
	#print(str(edge_list[index][0]))
	return edge_list[index][0], edge_list[index][1]


# This function extracts the edge pair in the form of matrices 
# Returns two matrices A and B
# A contains [a1x, a1y; a2x a2y] 
# B contains [b1x, b1y; b2x b2y]
def getEdgePairAsMatrix(X,i,j):
	A = np.zeros((2,2))
	B = np.zeros((2,2))
	
	i1, i2 = getNodesforEdge(i)
	j1, j2 = getNodesforEdge(j)

	A[0,:] = X[i1, :]
	A[1,:] = X[i2, :]

	B[0,:] = X[j1, :]
	B[1,:] = X[j2, :]

	return A,B

def num_crossings(G,X):
	num_intersections = 0

	# print X

	# print "Number of edges: ", m

	# loop through all edge pairs
	for i in range(0,m):
		for j in range(i+1,m):

			A,B = getEdgePairAsMatrix(X,i,j)
			# doIntersect(x11, y11, x12, y12, x21, y21, x22, y22)
			if(doIntersect(A[0][0], A[0][1], A[1][0], A[1][1], B[0][0], B[0][1], B[1][0], B[1][1])):
				
				# print A
				# print B
				# print i
				# print j
				
				num_intersections += 1

	return num_intersections


def plotGraphandStats(X):
	#print(X)
	#plt.scatter(X[:,0], X[:,1], color='red')

	# for every edge in the graph
	# draw a line joining the endpoints
	for i in range(0,m):
		i1, i2 = getNodesforEdge(i)
		A = np.zeros((2,2))

		A[0,:] = X[i1, :]
		A[1,:] = X[i2, :]
		#plt.plot(A[:,0] , A[:,1], color='blue')

	num_intersections = 0
	min_angle = math.pi/2.0

	#print("Number of edges: ", m)

	# loop through all edge pairs
	for i in range(0,m):
		for j in range(i+1,m):

			A,B = getEdgePairAsMatrix(X,i,j)
			# doIntersect(x11, y11, x12, y12, x21, y21, x22, y22)
			if(doIntersect(A[0][0], A[0][1], A[1][0], A[1][1], B[0][0], B[0][1], B[1][0], B[1][1])):
				
				# print A
				# print B
				# print i
				# print j
				x_pt, y_pt = getIntersection(A[0][0], A[0][1], A[1][0], A[1][1], B[0][0], B[0][1], B[1][0], B[1][1])
				theta = getAngleLineSeg(A[0][0], A[0][1], B[0][0], B[0][1], x_pt, y_pt)
				if theta > math.pi/2.0:
					theta = math.pi - theta
				if theta < min_angle:
					min_angle = theta

				num_intersections += 1

	#print("Number of Edge Crossings: ", num_intersections)
	#print("Minimum Angle: ", to_deg(min_angle))

	#plt.show()
	return to_deg(min_angle)


# This function computes the stress of an embedding. It takes as input the coordinates X, 
# weights (i.e. d_{ij}^(-2)), ideal distances between the nodes, and the number of nodes
# in the graph 

def stress(X, weights, distances, n):
	#global total_stress_time
	#start_time = time.time()
	# print "Parameters:", X, type(X)
	s = 0.0
	for i in range(0,n):
		for j in range(i+1,n):
			norm = X[i,:] - X[j,:]
			norm = np.sqrt(sum(pow(norm,2)))
			s += weights[i,j] * pow((norm - distances[i,j]), 2)
	#total_stress_time = total_stress_time + (time.time() - start_time)
	return s


# This function computes the stress of an embedding X. It needs weights, ideal distances, 
# and number of nodes already initialized
def stress_X(X):
	s = 0.0
	for i in range(0,n):
		for j in range(i+1,n):
			norm = X[i,:] - X[j,:]
			norm = np.sqrt(sum(pow(norm,2)))
			s += weights[i,j] * pow((norm - distances[i,j]), 2)
	return s


###### Initialize and load the graphs; Compute the weights and distances

#G = nx.petersen_graph()
# G = nx.complete_graph(6)
# G = nx.complete_graph(7)
# G = nx.wheel_graph(9)

#n = G.number_of_nodes()
#m = G.number_of_edges()
#edge_list = G.edges()

#distances = nx.floyd_warshall(G)

# Initialize the coordinates randomly in the range [-50, 50]
#X_curr = np.random.rand(n,2)*100 - 50

#if USE_NEATO_INITIAL:
#	pos = nx.nx_agraph.graphviz_layout(G)
	# Copy the coordinates from pos to X_curr
#	for i in range(0,n):
#		X_curr[i] = pos[i]


#plotGraphandStats(X_curr)

# Z=np.copy(X_curr)
#X_prev = np.copy(X_curr)

# Copy the distances into a 2D numpy array
#distances = np.array([[distances[i][j] for j in distances[i]] for i in distances])
# weights = 1/(d^2)
#weights = 1/pow(distances,2)
#weights[weights == inf] = 0


# Define: penalties, u_params, gammas, edgesID


# penalties: a 2D array containing the penalties for each possible edge pair
# For now the penalties start with 0 and gradually increase by 1 in the next iteration
# if the crossing persists.

#penalties = np.zeros((m, m))

# u_params: a 3D array containing the u vectors for each edge pair
#u_params = np.zeros((m, m, 2))

# gammas: a 2D array containing the gamma values for each possible edge pair
#gammas = np.zeros((m, m))

# all these variables need to be accessed as a 2D array
# with the edge pair as the i,j index of the 2D array.

#The 2D array is of size M*M where M is the number of edges. Max edges = n*(n-1)/2
# for complete graph


# This function computes the modified objective function i.e. a sum of stress and penalty function
def modified_cost(X, W, K):
	#global total_modified_cost_time
	#start_time = time.time()
	global NORMALIZE
	#Reshape the 1D array to a n*2 matrix
	X = X.reshape((n,2))
	return_val = 0.0
	if NORMALIZE==0:
		return_val = (W*stress(X, weights, distances, n)) + K*sum_penalty(X)
	else:
		s = stress(X, weights, distances, n)
		return_val = (W*s) + K*sum_penalty(X)/(m*m)
	#print('Time to compute modified_cost: ' + str((time.time() - start_time)) + ' seconds')
	#total_modified_cost_time = total_modified_cost_time + (time.time() - start_time)
	return return_val

def max_zero(a):
	return np.maximum(0,a)

def sum_penalty(X):
	#global total_sum_penalty_time, total_getEdgePairAsMatrix_time, total_getIntersection_time, total_getAngleLineSeg_time, total_just_sumPenalty_time, total_doIntersect_time
	#cdef int start_time = time.time()
	sumPenalty = 0

	#cdef int start_time_getEdgePairAsMatrix = 0
	#cdef int start_time_doIntersect = 0
	#cdef int start_time_getIntersection = 0
	#cdef int start_time_getAngleLineSeg = 0
	#cdef int start_time_just_sumPenalty = 0

	# loop through all edge pairs
	for i in range(0,m):
		for j in range(i+1,m):

			# Add the penalty
			# ||(-Au - eY)+||1 + ||(Bu + (1 + Y)e)+||1
			# z_+ = max(0,z)

			# sumPenalty += penalty_ij/2 * [||(-Ai(X)ui - Yie)+||1
			# + ||(Bi(X)ui + (1 + Yi)e)+||1] * cos^2(theta)

			#start_time_getEdgePairAsMatrix = time.time()
			A,B = getEdgePairAsMatrix(X,i,j)
			#total_getEdgePairAsMatrix_time = total_getEdgePairAsMatrix_time + (time.time() - start_time_getEdgePairAsMatrix)
			#start_time_doIntersect = time.time()
			intersecting = doIntersect(A[0][0], A[0][1], A[1][0], A[1][1], B[0][0], B[0][1], B[1][0], B[1][1])
			#total_doIntersect_time = total_doIntersect_time + (time.time() - start_time_doIntersect)
			if(intersecting):
				#start_time_getIntersection = time.time()
				x_pt, y_pt = getIntersection(A[0][0], A[0][1], A[1][0], A[1][1], B[0][0], B[0][1], B[1][0], B[1][1])
				#total_getIntersection_time = total_getIntersection_time + (time.time() - start_time_getIntersection)
				#start_time_getAngleLineSeg = time.time()
				theta = getAngleLineSeg(A[0][0], A[0][1], B[0][0], B[0][1], x_pt, y_pt)
				#total_getAngleLineSeg_time = total_getAngleLineSeg_time + (time.time() - start_time_getAngleLineSeg)
				#start_time_just_sumPenalty = time.time()
				sumPenalty += (penalties[i][j]/2.0) * (math.cos(theta)**2) * (np.sum(max_zero(-np.matmul(A,u_params[i][j])- gammas[i][j] * np.array([1,1]))) + np.sum(max_zero(np.matmul(B,u_params[i][j])+ (1+gammas[i][j]) * np.array([1,1]))))
				#total_just_sumPenalty_time = total_just_sumPenalty_time + (time.time() - start_time_just_sumPenalty)

	#total_sum_penalty_time = total_sum_penalty_time + (time.time() - start_time)
	return sumPenalty


def optimize(X_curr, wi, ki, W_param, K_param, G):
	# Start with pivotmds or neato stress majorization or cmdscale as in the paper
	# Or use X with a random initialization
	# Currently, we start with neato stress majorization coordinates

	# set penalty to 1 if there is an edge-crossing 
	# reset penalty to 0 if there is no edge-crossing

	global total_u_gamma_time, total_gradient_descent_time, total_penalties_after_grad_desc_time, W, K, X_new, NUMBER_OF_CROSSINGS
	W = W_param
	K = K_param

	X = np.copy(X_curr)

	start_time = time.time()
	# loop through all edge pairs
	for i in range(0,m):
		for j in range(i+1,m):
			A,B = getEdgePairAsMatrix(X,i,j)
			if(doIntersect(A[0][0], A[0][1], A[1][0], A[1][1], B[0][0], B[0][1], B[1][0], B[1][1])):
				# penalties[i][j] = penalties[i][j] + 1
				penalties[i][j] = 1
			else:
				penalties[i][j] = 0
	print('Time to count crossings at the beginning: ' + str((time.time() - start_time)) + ' seconds')

	#TODO: Be careful that the optimization does not monotonically decrease the cost function
	# This is basically one way to phrase "Unitl Satisfied" and is a very rigid way
	# Another way is to count the number of edge crossings in the graph
	# If the no. of edge crossings remains the same for a long time 
	# or the no. of edge crossings increases significantly
	# or the no. of edge crossings remains within a same range for a long time
	# then stop the optimization and store the embedding with the best edge crossing

	num_iters = 0
	while 1: 
		num_iters += 1

		# For all intersecting edge pairs
		# compute optimal u and gammas using the LP subroutine
		X = np.copy(X_curr)

		start_time = time.time()
		# loop through all edge pairs
		for i in range(0,m):
			for j in range(i+1,m):

				A,B = getEdgePairAsMatrix(X,i,j)
				# doIntersect(x11, y11, x12, y12, x21, y21, x22, y22)
				if(doIntersect(A[0][0], A[0][1], A[1][0], A[1][1], B[0][0], B[0][1], B[1][0], B[1][1])):
					
					# Call the LP module for optimization
					#Input to the module are two edges A and B
					# A is a 2*2 matrix that contains [a1x a1y; a2x a2y]
					# B is a 2*2 matrix that contains [b1x b1y; b2x b2y]
					# In the LP module, ax = a1x, ay = a1y, bx = a2x, by = a2y
					# cx = b1x, cy = b1y, dx = b2x, dy = b2y
					#u,gamma = LP_optimize(A,B)
					#u_params[i][j] = u
					
					# ux = get_ux(A[0][0], A[0][1], A[1][0], A[1][1], B[0][0], B[0][1], B[1][0], B[1][1])
					# uy = get_uy(A[0][0], A[0][1], A[1][0], A[1][1], B[0][0], B[0][1], B[1][0], B[1][1])
					# gamma = get_gamma(A[0][0], A[0][1], A[1][0], A[1][1], B[0][0], B[0][1], B[1][0], B[1][1])
					
					ux, uy, gamma = get_u_gamma(A[0][0], A[0][1], A[1][0], A[1][1], B[0][0], B[0][1], B[1][0], B[1][1])

					u_params[i][j][0] = ux
					u_params[i][j][1] = uy
					gammas[i][j] = gamma
		print('Time to determine ux, uy and gamma: ' + str((time.time() - start_time)) + 'seconds')
		total_u_gamma_time = total_u_gamma_time + (time.time() - start_time)

		# Use gradient descent to optimize the modified_cost function
		# keep the X as a flattened 1D array and reshape it inside the 
		# modified_cost function as a 2D array/matrix
		X = X.flatten()

		start_time = time.time()
		#res = minimize(modified_cost, X, args=(W, K), method='BFGS', jac=jacobian_Mod_Cost, options={'disp': True})

		if(THIS_GD_OPTION == 'VANILLA'):
			res = minimize_with_gd(X, num_iters=THIS_NUM_ITERS, alpha=THIS_ALPHA)
		elif(THIS_GD_OPTION == 'MOMENTUM'):
			res = minimize_with_momentum(X, num_iters=THIS_NUM_ITERS, alpha=THIS_ALPHA)
		elif(THIS_GD_OPTION == 'NESTEROV'):
			res = minimize_with_nesterov_momentum(X, num_iters=THIS_NUM_ITERS, alpha=THIS_ALPHA)
		elif(THIS_GD_OPTION == 'ADAGRAD'):
			res = minimize_with_adagrad(X, num_iters=THIS_NUM_ITERS, alpha=THIS_ALPHA)
		elif(THIS_GD_OPTION == 'RMSPROP'):
			res = minimize_with_rmsprop(X, num_iters=THIS_NUM_ITERS, alpha=THIS_ALPHA)
		elif(THIS_GD_OPTION == 'ADAM'):
			res = minimize_with_adam(X, num_iters=THIS_NUM_ITERS, alpha=THIS_ALPHA)

		print('Time to optimize using BFGS: ' + str((time.time() - start_time)) + ' seconds')
		total_gradient_descent_time = total_gradient_descent_time + (time.time() - start_time)
		X_prev = np.copy(X_curr)
		#X_curr = res.x.reshape((n,2))
		X_curr = res.reshape((n,2))
		print (str(W) + " " + str(K) + " " + str(num_iters))
		#print(res.x)


		#if IS_LOG_COST_FUNCTION:
		#	COST_FUNCTIONS[wi][ki][ii][num_iters-1] = modified_cost(res.x, W, K)

		X_new[wi][ki][num_iters-1] = X_curr
		NUMBER_OF_CROSSINGS[wi][ki][num_iters-1] = num_crossings(G,X_curr)

		X = np.copy(X_curr)

		# increase penalty by 1 if the crossing persists 
		# reset penalties to 0 if the crossing disappears

		start_time = time.time()
		# loop through all edge pairs
		for i in range(0,m):
			for j in range(i+1,m):
				A,B = getEdgePairAsMatrix(X,i,j)
				if(doIntersect(A[0][0], A[0][1], A[1][0], A[1][1], B[0][0], B[0][1], B[1][0], B[1][1])):
					# penalties[i][j] = penalties[i][j] + 1
					penalties[i][j] = 1
				else:
					penalties[i][j] = 0
		total_penalties_after_grad_desc_time = total_penalties_after_grad_desc_time + (time.time() - start_time)

		if (not USE_NUM_ITERS):
			if((modified_cost(X_prev) - modified_cost(X_curr)) / modified_cost(X_prev) < EPSILON):
				return X_curr
		else:
			if(num_iters >= NUM_ITERATIONS):
				return X_curr


# Construct the laplacian matrix of the weights
def constructLaplacianMatrix():
	L = -weights
	L[L==-inf] = 0
	diagL = np.diag(np.sum(weights, axis = 1))
	L = L + diagL
	return L

# This function computes the jacobian/gradient of the modified cost function at the point X
def jacobian_Mod_Cost(X):
	#Reshape the 1D array to a n*2 matrix
	global weights
	global distances
	global n
	global W
	global K
	X = X.reshape((n,2))
	dmodCost_dx = np.zeros((n,2))
	dmodCost_dx = dmodCost_dx + W*dStress_dx(X, weights, distances, n)
	dmodCost_dx = dmodCost_dx + K*dSumPenalty_dx(X)
	return dmodCost_dx.flatten()

# This function computes the gradient of stress function at the point X
def dStress_dx(X, weights, distances, n):

	dStress_dxArr = np.zeros((n,2))
	# for every node/point
	for i in range(0,n):
		# for every node's contribution to this node
		for j in range(0,n):
			diff = X[i,:] - X[j,:]
			# norm_diff = np.linalg.norm(diff)
			norm_diff = np.sqrt(sum(pow(diff,2)))
			if(norm_diff!=0):
				dStress_dxArr[i,:] += weights[i,j] * (norm_diff - distances[i,j]) * diff / norm_diff
		dStress_dxArr[i,:] *= 2

	return dStress_dxArr

# This function computes the gradient of sum penalty at the point X
def dSumPenalty_dx(X):

	dSumPenalty_dxArr = np.zeros((n,2))

	# Keep a list of all edge pair (indices) that intersect
	list_crossings = []
	# list_crossings.append([5,1])
	# list_crossings = np.array(list_crossings)

	# For each vertex, Keep a list of all edge pairs that intersect where 
	# the vertex is involved
	# Store the edge that intersects
	# Keep track of whether the edge falls to the left/right side of the edge pair
	vertex_crossing_info = {}

	# loop through all edge pairs
	for i in range(0,m):
		for j in range(i+1,m):

			A,B = getEdgePairAsMatrix(X,i,j)
			# doIntersect(x11, y11, x12, y12, x21, y21, x22, y22)
			if(doIntersect(A[0][0], A[0][1], A[1][0], A[1][1], B[0][0], B[0][1], B[1][0], B[1][1])):
				list_crossings.append([i,j])

				# for each vertex appearing in the edge pair
				# store the edge pair, the edge involved, and the left/right side on the edge pair
				i1, i2 = getNodesforEdge(i)
				j1, j2 = getNodesforEdge(j)

				if(not(i1 in vertex_crossing_info)):
					vertex_crossing_info[i1] = []
				if(not(i2 in vertex_crossing_info)):
					vertex_crossing_info[i2] = []
				if(not(j1 in vertex_crossing_info)):
					vertex_crossing_info[j1] = []
				if(not(j2 in vertex_crossing_info)):
					vertex_crossing_info[j2] = []

				i1_obj = {}
				i1_obj['edge_pair'] = [i,j]
				i1_obj['edge'] = i
				i1_obj['left'] = True

				i2_obj = {}
				i2_obj['edge_pair'] = [i,j]
				i2_obj['edge'] = i
				i2_obj['left'] = True

				j1_obj = {}
				j1_obj['edge_pair'] = [i,j]
				j1_obj['edge'] = j
				j1_obj['left'] = False

				j2_obj = {}
				j2_obj['edge_pair'] = [i,j]
				j2_obj['edge'] = j
				j2_obj['left'] = False

				vertex_crossing_info[i1].append(i1_obj)
				vertex_crossing_info[i2].append(i2_obj)
				vertex_crossing_info[j1].append(j1_obj)
				vertex_crossing_info[j2].append(j2_obj)

	# Convert the 2D list into 2D numpy array
	list_crossings = np.array(list_crossings)

	# Loop through the vertex crossings list
	# For each vertex 

	edge_pairs = []
	for node_index in vertex_crossing_info:
		edge_pairs = vertex_crossing_info[node_index]

	# For each contributing edge pair
	for edge_pairObj in edge_pairs:

		# Get the coordinates of the node
		node_coords = X[node_index, :]

		this_I = edge_pairObj['edge_pair'][0]
		this_J = edge_pairObj['edge_pair'][1]

		this_edge = edge_pairObj['edge']

		# Retrieve the u_params, gamma and penalty
		# this_ux = u_params[this_I][this_J][0]
		# this_uy = u_params[this_I][this_J][1]
		this_u = u_params[this_I][this_J]
		this_gamma = gammas[this_I][this_J]
		this_penalty = penalties[this_I][this_J]

		# Also include the penalty term in the gradient

		# If the vertex is on left side i.e. associated with -Au - gamma*e
			# if (-Au-gamma*e) > 0, add the derivative -ux, -uy for xi & yi
			# (-xi*ux - yi*uy - gamma > 0)

		if(edge_pairObj['left']):
			if(-np.dot(node_coords, this_u) - this_gamma > 0):
				dSumPenalty_dxArr[node_index, :] += (-this_u)*this_penalty/2.0

		# If the vertex is on right side i.e. associated with Bu + (1+gamma)*e
			# if (xi*ux + yi*uy + (1+gamma)) > 0, add the derivative
			# ux, uy for xi & yi
		else:
			if(np.dot(node_coords, this_u) + 1 + this_gamma > 0):
				dSumPenalty_dxArr[node_index, :] += (this_u)*this_penalty/2.0

	return dSumPenalty_dxArr

def minimize_with_gd(X, num_iters=100, beta1 = 0.9, beta2 = 0.999, alpha = 1e-3, eps = 1e-8):
        for t in range(1, num_iters+1):
                grad = jacobian_Mod_Cost(X)
                X -= alpha * grad

        return X

def minimize_with_momentum(X, num_iters=100, gamma = 0.9, alpha = 1e-3, eps = 1e-8):
        V = np.zeros(X.shape)

        for t in range(1, num_iters+1):
                grad = jacobian_Mod_Cost(X)
                V = gamma * V + alpha *grad
                X -= V

        return X

def minimize_with_nesterov_momentum(X, num_iters=100, gamma = 0.9, alpha = 1e-3, eps = 1e-8):
        V = np.zeros(X.shape)

        for t in range(1, num_iters+1):
                grad_ahead = jacobian_Mod_Cost(X + gamma*V)

                V = gamma * V + alpha *grad_ahead
                X -= V

        return X

def minimize_with_adagrad(X, num_iters=100, beta1 = 0.9, beta2 = 0.999, alpha = 1e-3, eps = 1e-8):
        R = np.zeros(X.shape)

        for t in range(1, num_iters+1):
                grad = jacobian_Mod_Cost(X)

                R += grad**2

                X -= alpha * grad / (np.sqrt(R) + eps)

        return X

def minimize_with_rmsprop(X, num_iters=100, gamma = 0.9, beta2 = 0.999, alpha = 1e-3, eps = 1e-8):
        R = np.zeros(X.shape)

        for t in range(1, num_iters+1):
                grad = jacobian_Mod_Cost(X)

                R = gamma*R + (1-gamma)*(grad**2)

                X -= alpha * grad / (np.sqrt(R) + eps)

        return X

def minimize_with_adam(X, num_iters=100, beta1 = 0.9, beta2 = 0.999, alpha = 1e-3, eps = 1e-8):
        M = np.zeros(X.shape)
        R = np.zeros(X.shape)

        for t in range(1, num_iters+1):
                grad = jacobian_Mod_Cost(X)

                M = beta1*M + (1-beta1)*grad
                R = beta2*R + (1-beta2)*(grad**2)

                m_hat = M / (1 - beta1**(t))
                r_hat = R / (1 - beta2**(t))

                X -= alpha * m_hat / (np.sqrt(r_hat) + eps)

        return X


def optimizer_jacob(output_folder, filename, W_start, W_end, K_start, K_end, layout_file_name, num_iterations, norm, OUTPUT_FILE_EXT, THIS_GD_OPTION_param, THIS_ALPHA_param, THIS_NUM_ITERS_param):
	global OUTPUT_FOLDER, FILENAME, USE_INITIAL, n, m, NORMALIZE, NUMBER_OF_CROSSINGS, COST_FUNCTIONS, X_new, init_X, total_u_gamma_time, NUM_ITERATIONS, total_gradient_descent_time, total_penalties_after_grad_desc_time, THIS_GD_OPTION, THIS_ALPHA, THIS_NUM_ITERS
	OUTPUT_FOLDER =output_folder
	FILENAME = filename
	G = build_networkx_graph(OUTPUT_FOLDER+'/'+FILENAME+'.txt')
	n = G.number_of_nodes()
	m = G.number_of_edges()
	NORMALIZE = norm
	NUM_ITERATIONS = num_iterations
	THIS_GD_OPTION = THIS_GD_OPTION_param
	THIS_ALPHA = THIS_ALPHA_param
	THIS_NUM_ITERS = THIS_NUM_ITERS_param

	NUMBER_OF_CROSSINGS = -np.ones((W_end-W_start, K_end-K_start, NUM_ITERATIONS));
	COST_FUNCTIONS = -np.ones((W_end-W_start, K_end-K_start, NUM_ITERATIONS));
	X_new = np.zeros((W_end-W_start, K_end-K_start, NUM_ITERATIONS, n, 2));
	init_X = np.zeros((W_end-W_start, K_end-K_start, n, 2));
	
	wi=0
	#for W in range(0,2):
	for W in range(W_start, W_end):
		ki=0
		#for K in [1, 10, 100, 1000, 10000]:
		for K in [math.pow(2,i) for i in range(K_start,K_end)]:
			USE_INITIAL = layout_file_name
			resultX = runOptimizer(G, W, K, wi, ki)
			#X_new[wi][ki][ii] = resultX
			#NUMBER_OF_CROSSINGS[wi][ki][ii]=num_crossings(G,resultX)

			ki = ki+1
		wi = wi+1

	# Write the graph into the dot file
	#write_dot(G, 'output/' + FILENAME + '.dot')
	write_networx_graph(G, 'output/' + FILENAME + '.txt')

	# Write all the other arrays into another file
	np.save(OUTPUT_FOLDER + '/' + FILENAME + '_ncr_'+OUTPUT_FILE_EXT, NUMBER_OF_CROSSINGS)
	np.save(OUTPUT_FOLDER + '/' + FILENAME + '_cost_'+OUTPUT_FILE_EXT, COST_FUNCTIONS)
	np.save(OUTPUT_FOLDER + '/' + FILENAME + '_xy_'+OUTPUT_FILE_EXT, X_new)
	np.save(OUTPUT_FOLDER + '/' + FILENAME + '_init_xy_'+OUTPUT_FILE_EXT, init_X)

	# To Read it back
	# np.load(fname + '.npy')

	print('total_u_gamma_time: '+str(total_u_gamma_time))
	print('total_gradient_descent_time: ' + str(total_gradient_descent_time))
	#print('total_stress_time: ' + str(total_stress_time))
	#print('total_sum_penalty_time: ' + str(total_sum_penalty_time))
	#print('total_modified_cost_time: ' + str(total_modified_cost_time))
	print('total_penalties_after_grad_desc_time: ' + str(total_penalties_after_grad_desc_time))
	#print('total_getEdgePairAsMatrix_time: ' + str(total_getEdgePairAsMatrix_time))
	#print('total_getIntersection_time: ' + str(total_getIntersection_time))
	#print('total_getAngleLineSeg_time: ' + str(total_getAngleLineSeg_time))
	#print('total_just_sumPenalty_time: ' + str(total_just_sumPenalty_time))
	#print('total_doIntersect_time: ' + str(total_doIntersect_time))


if len(sys.argv)<14:
	print('usage:python3 main_jacob.py OUTPUT_FOLDER file_prefix(er_10_0.6) W_start W_end K_start K_end layout_file_name num_iterations normalize(0/1) OUTPUT_FILE_EXT STRATEGY ALPHA GRADIENT_DESCENT_ITERATIONS')
	quit()
else:
	optimizer_jacob(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), sys.argv[7], int(sys.argv[8]), int(sys.argv[9]), sys.argv[10], sys.argv[11], float(sys.argv[12]), int(sys.argv[13]))



