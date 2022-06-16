from fa2 import ForceAtlas2
import networkx as nx
import numpy as np
import networkx.algorithms.community as nx_comm
from collections import defaultdict
import infomap
import beizer

# network visualization 
def force_pos(G):
    forceatlas2 = ForceAtlas2(
                          # Behavior alternatives
                          outboundAttractionDistribution=True,  # Dissuade hubs
                          linLogMode=False,  # NOT IMPLEMENTED
                          adjustSizes=False,  # Prevent overlap (NOT IMPLEMENTED)
                          edgeWeightInfluence=1.0,

                          # Performance
                          jitterTolerance=1.0,  # Tolerance
                          barnesHutOptimize=True,
                          barnesHutTheta=1.2,
                          multiThreaded=False,  # NOT IMPLEMENTED

                          # Tuning
                          scalingRatio=2.0,
                          strongGravityMode=False,
                          gravity=1.0,

                          # Log
                          verbose=True)
    pos=forceatlas2.forceatlas2_networkx_layout(G, pos=nx.spring_layout(G), iterations=2000)
    return pos

# Disparity filter 
# The implementation is based on the article:
# "Extracting the multiscale backbone of complex weighted networks" M. Ángeles Serrano, Marián Boguña, Alessandro Vespignani https://arxiv.org/pdf/0904.2389.pdf

# probability density function
def pdf(pij,k):
    return -(1-pij)**(k-1)-(-(1-0)**(k-1))   
# the sum of weights incident to the node
# normalize the weights
def s(G,i):
    neighbors = G.edges(i,data=True)
    return sum([node[2]['weight'] for node in neighbors]) 
def pij(G,i):
    ij=dict()
    k = len(G[i])
    if k > 1:
        for v in G[i]:
            w = G[i][v]['weight']
            p_ij = float(np.absolute(w))/s(G,i)
            ij[(i,v,w)]=p_ij
    return ij

def Disparity(G, a):
# filter implementation
# B:filtered graph
    B=nx.Graph()
    for i in G.nodes():
        k=G.degree(i)
        if k>1:
            pi=pij(G,i)
            for edges in pi.keys():
                aij=pdf(pi[edges],k)
                if 1-aij <a:
                    B.add_weighted_edges_from([edges])
    return B



# some functions for networks
# calculate modularity
def modularity(G, communities):
    v = defaultdict(list)
    for key, value in sorted(dict(communities).items()):
        v[value].append(key)
    return nx_comm.modularity(G, v.values())
#run infomap on weighted graph
def commu_detect_weighted(G):
    im = infomap.Infomap()
    links=[(i,j,float(k['weight'])) for i,j,k in list(G.edges(data=True))]
    im.add_links(links)
    im.run()
    communities = im.get_modules()
    n= im.num_top_modules
    m=modularity(G, communities)
    nx.set_node_attributes(G,communities,'community')
    return (n, m),G
def commu_detect(G):
    im = infomap.Infomap()
    links=[(i,j) for i,j in list(G.edges())]
    im.add_links(links)
    im.run()
    communities = im.get_modules()
    n= im.num_top_modules
    m=modularity(G, communities)
    nx.set_node_attributes(G,communities,'community')
    return (n, m),G



# curve edges 
# draw curve edges with python networkx

def curved_edges(G, selected_edges, pos, dist_ratio=0.2, bezier_precision=20, polarity='random'):
    # Get nodes into np array
    edges = np.array(selected_edges)
    l = edges.shape[0]

    if polarity == 'random':
        # Random polarity of curve
        rnd = np.where(np.random.randint(2, size=l)==0, -1, 1)
    else:
        # Create a fixed (hashed) polarity column in the case we use fixed polarity
        # This is useful, e.g., for animations
        rnd = np.where(np.mod(np.vectorize(hash)(edges[:,0])+np.vectorize(hash)(edges[:,1]),2)==0,-1,1)
    
    # Coordinates (x,y) of both nodes for each edge
    # e.g., https://stackoverflow.com/questions/16992713/translate-every-element-in-numpy-array-according-to-key
    # Note the np.vectorize method doesn't work for all node position dictionaries for some reason
    u, inv = np.unique(edges, return_inverse = True)
    coords = np.array([pos[x] for x in u])[inv].reshape([edges.shape[0], 2, edges.shape[1]])
    coords_node1 = coords[:,0,:]
    coords_node2 = coords[:,1,:]
    
    # Swap node1/node2 allocations to make sure the directionality works correctly
    should_swap = coords_node1[:,0] > coords_node2[:,0]
    coords_node1[should_swap], coords_node2[should_swap] = coords_node2[should_swap], coords_node1[should_swap]
    
    # Distance for control points
    dist = dist_ratio * np.sqrt(np.sum((coords_node1-coords_node2)**2, axis=1))

    # Gradients of line connecting node & perpendicular
    m1 = (coords_node2[:,1]-coords_node1[:,1])/(coords_node2[:,0]-coords_node1[:,0])
    m2 = -1/m1

    # Temporary points along the line which connects two nodes
    # e.g., https://math.stackexchange.com/questions/656500/given-a-point-slope-and-a-distance-along-that-slope-easily-find-a-second-p
    t1 = dist/np.sqrt(1+m1**2)
    v1 = np.array([np.ones(l),m1])
    coords_node1_displace = coords_node1 + (v1*t1).T
    coords_node2_displace = coords_node2 - (v1*t1).T

    # Control points, same distance but along perpendicular line
    # rnd gives the 'polarity' to determine which side of the line the curve should arc
    t2 = dist/np.sqrt(1+m2**2)
    v2 = np.array([np.ones(len(edges)),m2])
    coords_node1_ctrl = coords_node1_displace + (rnd*v2*t2).T
    coords_node2_ctrl = coords_node2_displace + (rnd*v2*t2).T

    # Combine all these four (x,y) columns into a 'node matrix'
    node_matrix = np.array([coords_node1, coords_node1_ctrl, coords_node2_ctrl, coords_node2])

    # Create the Bezier curves and store them in a list
    curveplots = []
    for i in range(l):
        nodes = node_matrix[:,i,:].T
        curveplots.append(bezier.Curve(nodes, degree=3).evaluate_multi(np.linspace(0,1,bezier_precision)).T)
      
    # Return an array of these curves
    curves = np.array(curveplots)
    return curves
