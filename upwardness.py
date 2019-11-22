import networkx as nx
import math


def y_distance(X, u, v):
    '''
    Returns the edge y-distance between the given vertices

    X: distance matrix
    u: the source
    v: the target
    return: the y - distance of the vertices (u,v)
    '''
    y_source = X[u][1]
    y_target = X[v][1]

    distance = math.abs(y_source - y_target)

    return distance

def is_upward_drawing(G, X):
  for e in G.edges():
    u, v = e
    if X[u][1] > X[v][1]:
      return False
  return True

def desired_y_distance(G, X, desired_distance=1):
    '''
    Computes the difference between the y-distance and the desired y-distance.
    G: the graph
    X: the distance matrix
    desired_distance: the desired y-distance (default=1)
    return: the error of the y-distance with respect the desired value
    '''

    edges = nx.edges(G)
    edge_count = len(edges)
    tot_error = 0.0

    for edge in edges:
        (s,t) = edge
        curr_distance = y_distance(X, s, t)
        distance_error = abs(curr_distance-desired_distance)
        tot_error += distance_error

    result = round(tot_error, 3)

    return result
