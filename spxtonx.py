import os
import sys

import networkx as nx
from networkx.drawing.nx_agraph import write_dot
from networkx.drawing.nx_agraph import read_dot as nx_read_dot


def writeSPXPositiontoNetworkXGraph(G, X):
    positions = dict()
    for v in nx.nodes(G):
        x = X[v, :][0]
        y = X[v, :][1]
        v_pos = str(x)+","+str(y)
        positions[v] = v_pos
    nx.set_node_attributes(G, positions, 'pos')
    return G
