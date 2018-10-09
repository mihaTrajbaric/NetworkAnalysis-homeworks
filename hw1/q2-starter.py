import snap
import numpy as np
import matplotlib.pyplot as plt



def load_graph(name):
    '''
    Helper function to load graphs.
    Wse "epinions" for Epinions graph and "email" for Email graph.
    Check that the respective .txt files are in the same folder as this script;
    if not, change the paths below as required.
    '''
    if name == "epinions":
        G = snap.LoadEdgeList(snap.PNGraph, "soc-Epinions1.txt", 0, 1)
    elif name == 'email':
        G = snap.LoadEdgeList(snap.PNGraph, "email-EuAll.txt", 0, 1)   
    else: 
        raise ValueError("Invalid graph: please use 'email' or 'epinions'.")
    return G

def q2_1_aux(name,id):
    G = load_graph(name)

    # Your code here:

    OutTreeEp = snap.GetBfsTree(G, id, True, False)
    InTreeEp = snap.GetBfsTree(G, id, False, True)
    sccOneRandNodeId = snap.GetMxScc(G).GetRndNId()

    sccInOutTree = OutTreeEp.IsNode(sccOneRandNodeId)
    sccInInTree = InTreeEp.IsNode(sccOneRandNodeId)
    print "graph:",name
    print "nodeId",id

    OutTree = snap.GetBfsTree(G, id, True, False)
    InTree = snap.GetBfsTree(G, id, False, True)
    sizeOutTree = OutTree.GetNodes()
    sizeInTree = InTree.GetNodes()
    print "sizegraph", G.GetNodes()
    print "sizeOutTree", sizeOutTree
    print "sizeInTree", sizeInTree

    if (sccInOutTree):
        if (sccInInTree):
            print "node in SCC"
        else:
            print "node in IN"
    else:
        print "node in OUT"
def q2_1():
    '''
    You will have to run the inward and outward BFS trees for the 
    respective nodes and reason about whether they are in SCC, IN or OUT.
    You may find the SNAP function GetBfsTree() to be useful here.
    '''
    
    ##########################################################################
    #TODO: Run outward and inward BFS trees from node 2018, compare sizes 
    #and comment on where node 2018 lies.

    q2_1_aux("email",2018)

    #node lies in IN

    ##########################################################################
    
    ##########################################################################
    #TODO: Run outward and inward BFS trees from node 224, compare sizes 
    #and comment on where node 224 lies.

    q2_1_aux("epinions", 224)
    #node lies in SCC
    
    
    
    
    ##########################################################################

    print '2.1: Done!\n'

def q2_2_aux(name):
    G = load_graph(name)

    vectOut = []
    vectIn = []
    for i in range(100):
        randInt = G.GetRndNId()

        OutTreeNodesN = snap.GetBfsTree(G, randInt, True, False).GetNodes()
        InTreeEpNodesN = snap.GetBfsTree(G, randInt, False, True).GetNodes()

        vectIn.append(InTreeEpNodesN)
        vectOut.append(OutTreeNodesN)
    vectOut.sort()
    vectIn.sort()
    return vectIn,vectOut

def plot(x_vect, y_vect, x_label, y_label, title, name):


    plt.semilogy(x_vect, y_vect, color='y', label=name)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.show()

def q2_2():
    '''
    For each graph, get 100 random nodes and find the number of nodes in their
    inward and outward BFS trees starting from each node. Plot the cumulative
    number of nodes reached in the BFS runs, similar to the graph shown in 
    Broder et al. (see Figure in handout). You will need to have 4 figures,
    one each for the inward and outward BFS for each of email and epinions.
    
    Note: You may find the SNAP function GetRndNId() useful to get random
    node IDs (for initializing BFS).
    '''
    ##########################################################################
    #TODO: See above.

    global emIn
    global emOut
    global epIn
    global epOut

    emIn, emOut = q2_2_aux("email")
    epIn, epOut = q2_2_aux("epinions")
    x_vect = [float(x)/100 for x in range (1,101)]
    #print x_vect
    plot(x_vect,epIn,"fract. of starting nodes","number of nodes reached","reachebility using inlinks ep","inlink reachibility")
    plot(x_vect,emIn,"fract. of starting nodes","number of nodes reached","reachebility using inlinks em","inlink reachibility")
    plot(x_vect,epOut,"fract. of starting nodes","number of nodes reached","reachebility using outlinks ep","outlink reachibility")
    plot(x_vect,emOut,"fract. of starting nodes","number of nodes reached","reachebility using outlinks em","outlink reachibility")

    ##########################################################################
    print '2.2: Done!\n'

def q2_3_aux(name):
    G = load_graph(name)

    SCC = snap.GetMxScc(G).GetNodes()
    wcc = snap.GetMxWcc(G).GetNodes()

    inexplosionVect = emIn if name == "email" else epIn
    outexplosionVect = emOut if name == "email" else epOut
    ineexpl = inexplosionVect[-1]
    outeexpl = outexplosionVect[-1]

    IN = ineexpl - SCC
    OUT = outeexpl - SCC

    DISCONNECTED = G.GetNodes()-wcc


    TENDRILS_AND_TUBES = wcc - IN - OUT - SCC
    print name,"DISCONNECTED:",DISCONNECTED,"IN:",IN,"OUT:",OUT,"SCC:",SCC,"TENDRILS + TUBES:",TENDRILS_AND_TUBES

    return
def q2_3():
    '''
    For each graph, determine the size of the following regions:
        DISCONNECTED
        IN
        OUT
        SCC
        TENDRILS + TUBES
        
    You can use SNAP functions GetMxWcc() and GetMxScc() to get the sizes of 
    the largest WCC and SCC on each graph. 
    '''
    ##########################################################################
    #TODO: See above.
    #Your code here:
    q2_3_aux("email")
    q2_3_aux("epinions")
    ##########################################################################
    print '2.3: Done!\n' 

def q2_4_aux(name):
    G = load_graph(name)
    counter = 0.0
    for i in range (1000):
        path = snap.GetShortPath(G, G.GetRndNId(), G.GetRndNId())

        if path != -1:
            counter += 1

    return counter / 1000
def q2_4():
    '''
    For each graph, calculate the probability that a path exists between
    two nodes chosen uniformly from the overall graph.
    You can do this by choosing a large number of pairs of random nodes
    and calculating the fraction of these pairs which are connected.
    The following SNAP functions may be of help: GetRndNId(), GetShortPath()
    '''
    ##########################################################################
    #TODO: See above.
    #Your code here:
    print "email probability of connectivity",q2_4_aux("email")
    print "epinions probability of connectivity",q2_4_aux("epinions")
    ##########################################################################
    print '2.4: Done!\n'
    
if __name__ == "__main__":
    q2_1()
    q2_2()
    q2_3()
    q2_4()
    print "Done with Question 2!\n"