import snap
import numpy as np
from itertools import permutations
from matplotlib import pyplot as plt

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
    plt.plot(np.linspace(0,10000,len(clustering_coeffs)), clustering_coeffs)
    plt.xlabel('Iteration')
    plt.ylabel('Average Clustering Coefficient')
    plt.title('Random edge rewiring: Clustering Coefficient')
    plt.savefig('q3_1.png', format='png')
    plt.show()

def gen_config_model_rewire(graph, iterations=10000):
    config_graph = graph
    clustering_coeffs = []
    ##########################################################################
    #TODO: Your code here



    ##########################################################################
    return config_graph, clustering_coeffs

def q3_1():
    '''
    Main q3 workflow. All of the work can be done in gen_config_model_rewire
    but you may modify this function as needed.
    '''
    G = load_graph("grid")
    config_graph, clustering_coeffs = gen_config_model_rewire(G, 10000)
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
    motif_counts = [0]*len(directed_3) # Reset the motif counts (Do not remove)
    ##########################################################################
    #TODO: Your code here



    ##########################################################################


def extend_subgraph(G, k, sg, v_ext, node_id, verbose=False):
    '''
    This is the recursive function in the ESU algorithm
    The base case is already implemented and calls count_iso. You should not
    need to modify this.

    Implement the recursive case.
    '''
    # Base case (you should not need to modify this):
    if len(sg) is k:
        count_iso(G, sg, verbose)
        return
    # Recursive step:
    ##########################################################################
    #TODO: Your code here



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
    G = load_graph("grid")
    #TODO: Your code here



    # Experiment for the email dataset
    G = load_graph("email")
    #TODO: Your code here



    ##########################################################################

if __name__ == "__main__":
    # Two global variables. Do not modify.
    directed_3 = load_3_subgraphs()
    motif_counts = [0]*len(directed_3)
    # Questions
    q3_1()
    #q3_2()
    #q3_3()
    print "Done with Question 3!\n"
