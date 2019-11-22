import networkx as nx
import math

def edge_length(X, u, v):
    '''
    Returns the edge length between the given vertices

    X: distance matrix
    u: the source
    v: the target
    return: the length of the edge (u,v)
    '''

    x_source = X[u][0]
    y_source = X[u][1]
    x_target = X[v][0]
    y_target = X[v][1]

    length = math.sqrt((x_source - x_target)**2 + (y_source - y_target)**2)

    return length



def avg_edge_length(edges, X):
    '''
    Computes the average edge length of the given graph layout
    edges: the edges of the graph
    X: distance matrix
    returns: the average edge length
    '''

    sum_edge_length = 0.0

    for edge in edges:
        (s,t) = edge
        curr_length = edge_length(X, s, t)
        sum_edge_length += curr_length

    edges_count = len(edges)
    avg_edge_len = sum_edge_length/edges_count

    return avg_edge_len

def desired_edge_length(G, X, desired_length=100):
    '''
    Computes the difference between the edge lengths and the desired length.
    G: the graph
    X: the distance matrix
    desired_length: the desired edge length (default=100)
    return: the error of the edge lengtsh with respect the given value
    '''

    edges = nx.edges(G)
    edge_count = len(edges)
    tot_error = 0.0

    for edge in edges:
        (s,t) = edge
        curr_length = edge_length(X, s, t)
        length_error = abs(curr_length-desired_length)
        tot_error += length_error

    result = round(tot_error, 3)

    return result


def uniformity_edge_length(G, X):
    '''
    The Edge length uniformity corresponds to the normalized standard deviation of the edge length.
    G: the graph
    X: the distance matrix
    return: the edge length uniformity of G rounded to the 3rd digit
    '''

    edges = nx.edges(G)
    edge_count = len(edges)
    avgEdgeLength = avg_edge_length(edges, X)
    tot_sum = 0.0

    for edge in edges:

        (s,t) = edge
        curr_length = edge_length(X, s, t)
        sum_edge_length += curr_length

        num = (curr_length-avgEdgeLength)**2
        den = edge_count*(avgEdgeLength**2)

        currValue = num/den
        tot_sum += currValue

    uniformity_e_len = math.sqrt(tot_sum)

    result = round(uniformity_e_len, 3)

    return result
