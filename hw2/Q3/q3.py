from __future__ import division
import snap
import numpy as np
from itertools import permutations
from matplotlib import pyplot as plt



def getNeighbours(graph,node_id):
    """

    :param graph: directed graph (PNGraph)
    :param node_id: id of a node
    :return: list of nbr node ids
    """
    nbr_list = []
    node = graph.GetNI(node_id)
    node_degree = node.GetDeg()
    for i in range (node_degree):
        nbr_id = node.GetNbrNId(i)
        nbr_list.append(nbr_id)
    return nbr_list

def load_graph(name):
    '''
    Helper function to load graphs.
    Check that the respective .txt files are in the same folder as this script;
    if not, change the paths below as required.
    '''
    if name == "email":
        G = snap.LoadEdgeList(snap.PNGraph, "email-Eu-core.txt", 0, 1)
    elif name == 'grid':
        G = snap.LoadEdgeList(snap.PNGraph, "USpowergrid_n4941.txt", 0, 1)
    elif name == 'esu_test':
        G = snap.LoadEdgeList(snap.PNGraph, "esu_test.txt", 0, 1)
    else:
        raise ValueError("Invalid graph: please use 'email' 'grid' or 'esu_test'.")
    return G

def load_3_subgraphs():
    '''
    Loads a list of all 13 directed 3-subgraphs.
    The list is in the same order as the figure in the HW pdf, but it is
    zero-indexed
    '''
    return [snap.LoadEdgeList(snap.PNGraph, "./subgraphs/{}.txt".format(i), 0, 1) for i in range(13)]

def plot_q3_1(clustering_coeffs):
    '''
    Helper plotting code for question 3.1 Feel free to modify as needed.
    '''
    plt.plot(np.linspace(0,8000,len(clustering_coeffs)), clustering_coeffs)
    plt.xlabel('Iteration')
    plt.ylabel('Average Clustering Coefficient')
    plt.title('Random edge rewiring: Clustering Coefficient')
    plt.savefig('q3_1.png', format='png')
    plt.show()

def gen_config_model_rewire(graph, iterations=8000, seed = 42):
    config_graph = graph
    clustering_coeffs = []
    ##########################################################################


    nOfNodes = graph.GetNodes()
    nOfEdges = graph.GetEdges()
    for j in range (iterations):
        while True:
            edges1 = graph.BegEI()
            eId = int(round(np.random.rand() * (nOfEdges-1)))
            for i in range(eId):
                edges1.Next()
            e1 = edges1
            a = e1.GetSrcNId()
            b = e1.GetDstNId()

            edges2 = graph.BegEI()
            e2id = int(round(np.random.rand() * (nOfEdges-1)))
            for i in range(e2id):
                edges2.Next()
            e2 = edges2

            c = e2.GetSrcNId()
            d = e2.GetDstNId()

            u = a if np.random.rand() > 0.5 else b
            v = b if u == a else a
            w = c if np.random.rand() > 0.5 else d
            x = d if w == c else c

            # check if graph is regular
            if u == w or v == x or (a == b and c == d) or graph.IsEdge(u, w) or graph.IsEdge(v, x):
                continue
            graph.DelEdge(a, b)
            graph.DelEdge(c, d)

            graph.AddEdge(u, w)
            graph.AddEdge(v, x)
            break
        if graph.GetNodes() != nOfNodes or graph.GetEdges() != nOfEdges:
            print "Graph has changed! nodes:",nOfNodes,"-->",graph.GetNodes(),"edges:",nOfEdges,"-->",graph.GetEdges()
            raise RuntimeError
        if j%100 == 0:
            clustering_coeffs.append(snap.GetClustCfAll(graph, snap.TFltPrV())[0])





    ##########################################################################
    return config_graph, clustering_coeffs

def q3_1():
    '''
    Main q3 workflow. All of the work can be done in gen_config_model_rewire
    but you may modify this function as needed.
    '''
    G = load_graph("grid")
    config_graph, clustering_coeffs = gen_config_model_rewire(G, 8000)
    plot_q3_1(clustering_coeffs)

def match(G1, G2):
    '''
    This function compares two graphs of size 3 (number of nodes)
    and checks if they are isomorphic.
    It returns a boolean indicating whether or not they are isomorphic
    You should not need to modify it, but it is also not very elegant...
    '''
    if G1.GetEdges() > G2.GetEdges():
        G = G1
        H = G2
    else:
        G = G2
        H = G1
    # Only checks 6 permutations, since k = 3
    for p in permutations(range(3)):
        edge = G.BegEI()
        matches = True
        while edge < G.EndEI():
            if not H.IsEdge(p[edge.GetSrcNId()], p[edge.GetDstNId()]):
                matches = False
                break
            edge.Next()
        if matches:
            break
    return matches

def count_iso(G, sg, verbose=False):
    '''
    Given a set of 3 node indices in sg, obtains the subgraph from the
    original graph and renumbers the nodes from 0 to 2.
    It then matches this graph with one of the 13 graphs in
    directed_3.
    When it finds a match, it increments the motif_counts by 1 in the relevant
    index

    IMPORTANT: counts are stored in global motif_counts variable.
    It is reset at the beginning of the enumerate_subgraph method.
    '''
    if verbose:
        print(sg)
    nodes = snap.TIntV()
    for NId in sg:
        nodes.Add(NId)
    # This call requires latest version of snap (4.1.0)
    SG = snap.GetSubGraphRenumber(G, nodes)
    for i in range(len(directed_3)):
        if match(directed_3[i], SG):
            motif_counts[i] += 1

def enumerate_subgraph(G, k=3, verbose=False):
    '''
    This is the main function of the ESU algorithm.
    Here, you should iterate over all nodes in the graph,
    find their neighbors with ID greater than the current node
    and issue the recursive call to extend_subgraph in each iteration

    A good idea would be to print a progress report on the cycle over nodes,
    So you get an idea of how long the algorithm needs to run
    '''
    global motif_counts
    motif_counts = [0]*len(directed_3) # Reset the motif counts (Do not remove)
    ##########################################################################
    size_nodes = G.GetNodes()
    counter = 0
    for node in G.Nodes():
        counter += 1
        node_id = node.GetId()
        v_ext = set([x_id for x_id in getNeighbours(G,node_id) if x_id > node_id ])
        v_sg = {node_id}
        extend_subgraph(G,k,v_sg,v_ext,node_id,verbose)
        #report print
        if counter%100 == 0:
            print counter / size_nodes * 100, "%"
    ##########################################################################
def n_excl(G,w,v_sg):
    """

    :param w: node
    :param v_sg: subgraph
    :return: set of neighbors of w that are not in subgraph or immidiate neighbors of subgraph nodes
    """
    w_nbr_set = set(getNeighbours(G, w))
    v_sg_nbr = set()
    for x in v_sg:
        v_sg_nbr = v_sg_nbr | set(getNeighbours(G, x))
    n_excl_set = w_nbr_set - (v_sg | v_sg_nbr)
    return n_excl_set

def extend_subgraph(G, k, v_sg, v_ext, v_node_id, verbose=False):
    """

    :param G: PNGraph
    :param k: size of the subgraph
    :param v_sg: current subgraph (node ids)
    :param v_ext: set of vertibces to extend with (node ids)
    :param v_node_id: id of current node
    :param verbose: print?
    :return: none
    """
    # Base case (you should not need to modify this):
    if len(v_sg) is k:
        count_iso(G, v_sg, verbose)
        return
    # Recursive step:
    ##########################################################################
    while len(v_ext) != 0:
        w = v_ext.pop()
        n_excl_set = set( u for u in n_excl(G, w, v_sg) if u > v_node_id)
        v_ext_prime = v_ext | n_excl_set
        extend_subgraph(G, k, (v_sg | {w}), v_ext_prime, v_node_id, verbose)
    ##########################################################################

def q3_2(verbose=False):
    '''
    This is all you really need to do for q2! Just set verbose to True to
    print the subgraphs and the reulting motif counts
    '''
    G = load_graph("esu_test")
    enumerate_subgraph(G, 3, verbose)
    if verbose:
        print(motif_counts)

def q3_3_aux(graph_name):
    motifs_grid = np.zeros((10, 13))
    for i in range(10):
        G = load_graph(graph_name)
        print "generating", i, "th null model"
        model, clustering = gen_config_model_rewire(G)
        print "enumerating subgraph"
        enumerate_subgraph(model)
        motifs_grid[i, :] = motif_counts
    for line in motifs_grid:
        print line
    G = load_graph("grid")
    enumerate_subgraph(G)
    # v motifs_grid imamo sedaj score za originalen graf
    # zdej mormo pa kej zracunat
    motifs_grid_mean = np.mean(motifs_grid, axis=0)
    motifs_grid_standard = np.std(motifs_grid, axis=0)
    zScores_grid = np.divide(np.subtract(motif_counts, motifs_grid_mean), motifs_grid_standard)
    print graph_name,"z scores",zScores_grid.tolist()

def q3_3():
    '''
    Here you should implement question 3.3
    You may initialize the np array with
        motifs = np.zeros((10,13))
    and assign to it with
        motifs[i,:] = motif_counts
    '''
    ##########################################################################
    # Experiment for the Power grid dataset
    #print("grid")
    #q3_3_aux("grid")
    print("email")
    q3_3_aux("email")

    ##########################################################################

if __name__ == "__main__":
    # we seed numpy with
    np.random.seed(42)
    # And now for snap
    Rnd = snap.TRnd(42)  # globally, at the beginning of your script
    # And then pass it every call to GetRndNId
    # Two global variables. Do not modify.
    directed_3 = load_3_subgraphs()
    motif_counts = [0]*len(directed_3)
    # Questions
    #q3_1()
    #q3_2(True)
    q3_3()
    print "Done with Question 3!\n"
