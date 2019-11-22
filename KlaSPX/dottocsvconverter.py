import os
import sys

import networkx as nx
from networkx.drawing.nx_agraph import write_dot
from networkx.drawing.nx_agraph import read_dot as nx_read_dot


def convert(G, outputgraphfile, outputlayoutfile):

    v_pos = nx.get_node_attributes(G, 'pos')

    v_id_map = dict()

    sorted_keys = sorted(v_pos.keys())
    # print(sorted_keys)
    id_new = 0

    pos_str = ""
    edges_str = ""

    for curr_key in sorted_keys:
        v_id_map[curr_key] = id_new
        id_new += 1
        pos_str += v_pos[curr_key] + "\n"
        # print(curr_key, id_new, v_pos[curr_key])

    for e in nx.edges(G):

        (u, v) = e

        u_new = v_id_map[u]
        v_new = v_id_map[v]


        edges_str += str(u_new) + "," + str(v_new) + "\n"


    v_file = open(outputlayoutfile, "w")
    e_file = open(outputgraphfile, "w")

    v_file.writelines(pos_str)
    e_file.writelines(edges_str)

    v_file.close()
    e_file.close()

    return
