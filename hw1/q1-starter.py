################################################################################
# CS 224W (Fall 2018) - HW1
# Starter code for Problem 1
# Author: praty@stanford.edu
# Last Updated: Sep 27, 2018
################################################################################

import snap
import numpy as np
import matplotlib.pyplot as plt
import random

# Setup
erdosRenyi = None
smallWorld = None
collabNet = None



# Problem 1.1
def genErdosRenyi(N=5242, E=14484):
    """
    :param - N: number of nodes
    :param - E: number of edges

    return type: snap.PUNGraph
    return: Erdos-Renyi graph with N nodes and E edges
    """
    ############################################################################
    # TODO: Your code here!
    Graph = snap.TUNGraph.New()
    for i in range (N):
    	Graph.AddNode(i)
    for i in range (E):
    	index = -2
    	while (index!=-1):
    		index = Graph.AddEdge(random.randint(0,N-1), random.randint(0,N-1))


    ############################################################################
    return Graph


def genCircle(N=5242):
    """
    :param - N: number of nodes

    return type: snap.PUNGraph
    return: Circle graph with N nodes and N edges. Imagine the nodes form a
        circle and each node is connected to its two direct neighbors.
    """
    ############################################################################
    # TODO: Your code here!
    Graph = snap.TUNGraph.New()

    for i in range (N):
    	Graph.AddNode(i)
    for i in range (N-1):
    	Graph.AddEdge(i,i+1)
    Graph.AddEdge(0,N-1)
    ############################################################################
    return Graph


def connectNbrOfNbr(Graph, N=5242):
    """
    :param - Graph: snap.PUNGraph object representing a circle graph on N nodes
    :param - N: number of nodes

    return type: snap.PUNGraph
    return: Graph object with additional N edges added by connecting each node
        to the neighbors of its neighbors
    """
    ############################################################################
    # TODO: Your code here!
    for i in range (N-2):
    	Graph.AddEdge(i,i+2)
    Graph.AddEdge(0,N-2)
    Graph.AddEdge(1,N-1)

    ############################################################################
    return Graph


def connectRandomNodes(Graph, M=4000):
    """
    :param - Graph: snap.PUNGraph object representing an undirected graph
    :param - M: number of edges to be added

    return type: snap.PUNGraph
    return: Graph object with additional M edges added by connecting M randomly
        selected pairs of nodes not already connected.
    """
    ############################################################################
    # TODO: Your code here!
    N = Graph.GetNodes()
    for i in range (M):
    	index = -2
    	while (index!=-1):
    		index = Graph.AddEdge(random.randint(0,N-1), random.randint(0,N-1))

    ############################################################################
    return Graph


def genSmallWorld(N=5242, E=14484):
    """
    :param - N: number of nodes
    :param - E: number of edges

    return type: snap.PUNGraph
    return: Small-World graph with N nodes and E edges
    """
    Graph = genCircle(N)
    Graph = connectNbrOfNbr(Graph, N)
    Graph = connectRandomNodes(Graph, 4000)
    return Graph


def loadCollabNet(path):
    """
    :param - path: path to edge list file

    return type: snap.PUNGraph
    return: Graph loaded from edge list at `path and self edges removed

    Do not forget to remove the self edges!
    """
    ############################################################################
    # TODO: Your code here!

    Graph = snap.LoadEdgeList(snap.PUNGraph, path, 0, 1)

    snap.DelSelfEdges(Graph)

  

    ############################################################################
    return Graph


def getDataPointsToPlot(Graph):
    """
    :param - Graph: snap.PUNGraph object representing an undirected graph

    return values:
    X: list of degrees
    Y: list of frequencies: Y[i] = fraction of nodes with degree X[i]
    """
    ############################################################################
    # TODO: Your code here!
    
    X, Y = [], []
    DegToCntV = snap.TIntPrV()
    snap.GetDegCnt(Graph, DegToCntV)	
    for p in DegToCntV:
        X.append(p.GetVal1())
        Y.append(p.GetVal2())
    ############################################################################
    return X, Y


def Q1_1():
    """
    Code for HW1 Q1.1
    """
    global erdosRenyi, smallWorld, collabNet
    erdosRenyi = genErdosRenyi(5242, 14484)
    smallWorld = genSmallWorld(5242, 14484)
    collabNet = loadCollabNet("ca-GrQc.txt")

    x_erdosRenyi, y_erdosRenyi = getDataPointsToPlot(erdosRenyi)
    plt.loglog(x_erdosRenyi, y_erdosRenyi, color = 'y', label = 'Erdos Renyi Network')

    x_smallWorld, y_smallWorld = getDataPointsToPlot(smallWorld)
    plt.loglog(x_smallWorld, y_smallWorld, linestyle = 'dashed', color = 'r', label = 'Small World Network')

    x_collabNet, y_collabNet = getDataPointsToPlot(collabNet)
    plt.loglog(x_collabNet, y_collabNet, linestyle = 'dotted', color = 'b', label = 'Collaboration Network')

    plt.xlabel('Node Degree (log)')
    plt.ylabel('Proportion of Nodes with a Given Degree (log)')
    plt.title('Degree Distribution of Erdos Renyi, Small World, and Collaboration Networks')
    plt.legend()
    plt.show()


# Execute code for Q1.1
Q1_1()


# Problem 1.2 - Clustering Coefficient

def calcClusteringCoefficientSingleNode(Node, Graph):
    """
    :param - Node: node from snap.PUNGraph object. Graph.Nodes() will give an
                   iterable of nodes in a graph
    :param - Graph: snap.PUNGraph object representing an undirected graph

    return type: float
    returns: local clustering coeffient of Node
    """
    ############################################################################
    # TODO: Your code here!
    degNode = Node.GetDeg()
    if (degNode < 2):
    	return 0.0
    #find out number of edges

    counter = 0.0
    neighbours = []
    #neighboursSet = set()

    for i in range (degNode):
    	neighbours.append(Node.GetNbrNId(i))
    	#neighboursSet.add(Node.GetNbrNId(i))
    neighboursSet = set(neighbours)

    #print neighbours
    #print neighboursSet

    for nbrId in neighbours:

    	nbrNode = Graph.GetNI(nbrId)
    	deg = nbrNode.GetDeg()
    	#print "nodeId, deg"
    	#print nbrId,deg
    	for i in range (deg):
    		nbrnbrId = nbrNode.GetNbrNId(i)
    		#print "      neighbour, in set"
    		#print "     ",nbrnbrId,(nbrnbrId in neighboursSet)
    		if (nbrnbrId in neighboursSet):
    			counter = counter + 1

		#print neighboursSet
    	neighboursSet.remove(nbrId)

    C = 2 * counter / (degNode * (degNode-1))

    ############################################################################
    return C

def calcClusteringCoefficient(Graph):
    """
    :param - Graph: snap.PUNGraph object representing an undirected graph

    return type: float
    returns: clustering coeffient of Graph
    """
    ############################################################################
    # TODO: Your code here! If you filled out calcClusteringCoefficientSingleNode,
    #       you'll probably want to call it in a loop here
    Coef = 0.0
    for Node in Graph.Nodes():
    	Coef = Coef + calcClusteringCoefficientSingleNode(Node, Graph)
    C = Coef / Graph.GetNodes()

    ############################################################################
    return C

def Q1_2():
    """
    Code for Q1.2
    """
    C_erdosRenyi = calcClusteringCoefficient(erdosRenyi)
    C_smallWorld = calcClusteringCoefficient(smallWorld)
    C_collabNet = calcClusteringCoefficient(collabNet)

    print('Clustering Coefficient for Erdos Renyi Network: %f' % C_erdosRenyi)
    print('Clustering Coefficient for Small World Network: %f' % C_smallWorld)
    print('Clustering Coefficient for Collaboration Network: %f' % C_collabNet)

    #print('Clustering Coefficient official for Erdos Renyi Network: %f' % snap.GetClustCf(erdosRenyi))
    #print('Clustering Coefficient official for Small World Network: %f' % snap.GetClustCf(smallWorld))
    #print('Clustering Coefficient official for Collaboration Network: %f' % snap.GetClustCf(collabNet))


# Execute code for Q1.2
Q1_2()

