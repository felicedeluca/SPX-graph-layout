# format_for_klapaukh.py
#
# Input: graphfile layoutfile scale [outfile]
#        Where the graph and layout are stored as SVGs
#
# Output: An SVG with an adjacency matrix and node locations
#         padded with a bunch of boilerplate for an experiment
#         on force-directed layout that Klapaukh was running,
#         but we are not.

import os
import sys

import networkx as nx


##########
# PARSER #
##########

# def parse_args(argv):
# 	scale = 1
# 	if len(argv) < 3:
# 		print("Usage: {0} graphfile layoutfile scale [outfile]".format(argv[0]))
# 		exit()
# 	g_file = argv[1]
# 	l_file = argv[2]
# 	scale = int(argv[3])
# 	if len(argv) > 4:
# 		outfile = argv[4]
# 	else:
# 		outfile = None
# 	return g_file, l_file, scale, outfile



########
# MATH #
########

def scale_layout(layout, scale):
	out = []
	for pt in layout:
		out.append(map(lambda x: (x+1)/2*scale, pt))
	return out

# def connected(i,j,edges):
# 	if i == j:
# 		return False
# 	for edge in edges:
# 		if((i in edge) and (j in edge)) or ((str(i) in edge) and (str(j) in edge)):
# 			return True
# 	return False

###################
# STRING BUILDING #
###################

def generate_boilerplate(w=800,h=800):
	out = """<?xml version="1.0" encoding="ISO-8859-1" standalone="no"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 20010904//EN"
"http://www.w3.org/TR/2001/REC-SVG-2010904/DTD/svg10.dtd">
<svg xmlns="http://www.w3.org/2000/svg"
xmlns:xlink="http://www.w3.org/1999/xlink" xml:space="preserve"
width="{0}"
height="{1}"
viewBox = "0 0 {0} {1}"
zoomAndPan="disable">
<!--
elapsed: 1
filename: test.xml
width: {0}
height: {1}
iterations: 1
forcemode: 1
ke: 0
kh: 0
kl: 0
kw: 0
mass: 0
time: 0
coefficientOfRestitution: 0
mus: 0
muk: 0
kg: 0
wellMass: 0
edgeCharge: 0
finalKineticEnergy: 0
nodeWidth: 0
nodeHeight: 0
nodeCharge: 0
-
Start Graph:
""".format(w,h)
	return out


def format_layout_data_networkx(G):

	nodes = sorted(nx.nodes(G))
	edges = nx.edges(G)
	all_pos = nx.get_node_attributes(G, 'pos')
	# print(all_pos)

	out = str(len(nodes)) + "\n"

	for i in range(0, len(nodes)):

		source = nodes[i]
		curr_node_x = all_pos[source].split(",")[0]
		curr_node_y = all_pos[source].split(",")[1]

		line = str(curr_node_x) + " " + str(curr_node_y)

		for j in range(0, len(nodes)):
			target = nodes[j]
			if i == j:
				line += " 0"
				continue

			if ((source,target) in edges) or ((target,source) in edges):
				line += " 1"
			else:
				line += " 0"

		out += line + "\n"
	return out


# def format_layout_data(layout, edges):
# 	out = str(len(layout)) + "\n"
# 	for i in range(0, len(layout)):
#
# 		curr_pos = list(layout[i])
# 		line = str(curr_pos[0]) + " " + str(curr_pos[1])
# 		for j in range(0, len(layout)):
# 			if i == j:
# 				line += " 0"
# 				continue
# 			if connected(i, j, edges):
# 				line += " 1"
# 			else:
# 				line += " 0"
#
# 		out += line + "\n"
# 	return out


# def build_svg(layout, edges, scale):
# 	svg = generate_boilerplate(scale, scale)
# 	svg += format_layout_data(layout, edges)
# 	svg += "</svg>"
# 	return svg

def build_svg_networkx(G, scale):
	svg = generate_boilerplate(scale, scale)
	svg += format_layout_data_networkx(G)
	svg += "</svg>"
	# print(svg)
	return svg



# ############
# # FILE I/O #
# ############
#
# def load_csv(filename, typ=""):
# 	data = list()
# 	with open(filename, 'r') as f:
# 		lines = f.readlines()
# 	for line in lines:
# 		entry = line.strip().split(',')
# 		# Ignore blank lines
# 		if entry[0] == '':
# 			continue
# 		if typ == "int":
# 			entry = map(int,entry)
# 		elif typ == "float":
# 			entry = map(float,entry)
# 		entry = list(entry)
# 		data.append(list(entry))
# 	return data


# def load_graph(g_file, l_file):
# 	layout = load_csv(l_file, "float")
# 	edges = load_csv(g_file, "int")
# 	return layout, edges

def write_svg(filename, svg):
	if filename != None:
		with open(filename, 'w') as f:
			f.write(svg)
	else:
		print(svg)
