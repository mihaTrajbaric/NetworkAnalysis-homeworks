import snap
import numpy as np
from matplotlib import pyplot as plt

def load_graph(name):
    '''
    Helper function to load undirected graphs.
    Check that the respective .txt files are in the same folder as this script;
    if not, change the paths below as required.
    '''
    if name == "normal":
        G = snap.LoadEdgeList(snap.PUNGraph, "polblogs.txt", 0, 1)
    elif name == 'rewired':
        G = snap.LoadEdgeList(snap.PUNGraph, "polblogs-rewired.txt", 0, 1)
    elif name == 'sample':
        G = snap.LoadEdgeList(snap.PUNGraph, "q1-sample.txt", 0, 1)
    else:
        raise ValueError("Invalid graph: please use 'normal', 'rewired' or 'sample'.")
    return G

def get_adjacency_matrix(Graph):
        '''
    This function might be useful for you to build the adjacency matrix of a
    given graph and return it as a numpy array
    '''
    ##########################################################################
    #TODO: Your code here



    return
    ##########################################################################

def get_sparse_degree_matrix(Graph):
    '''
    This function might be useful for you to build the degree matrix of a
    given graph and return it as a numpy array
    '''
    ##########################################################################
    #TODO: Your code here



    return
    ##########################################################################

def normalized_cut_minimization(Graph):
    '''
    Implement the normalized cut minimizaton algorithm we derived in the last
    homework here
    '''
    A = get_adjacency_matrix(Graph)
    D = get_sparse_degree_matrix(Graph)
    ##########################################################################
    #TODO: Your code here



    return
    ##########################################################################

def modularity(Graph, c1, c2):
    '''
    This function might be useful to compute the modularity of a given cut
    defined by two sets c1 and c2. We would normally require sets c1 and c2
    to be disjoint and to include all nodes in Graph
    '''
    ##########################################################################
    #TODO: Your code here



    return
    ##########################################################################

def q1_1():
    '''
    Main q1_1 workflow. All of the work can be done in gen_config_model_rewire
    but you may modify this function as needed.
    '''
    ##########################################################################
    #TODO: Your code here



    pass
    ##########################################################################

def SSBM(n,pi,a,b):
    '''

    '''
    ##########################################################################
    #TODO: Your code here



    pass
    ##########################################################################

def q1_2():
    '''
    You can probably just implement everything required for question 1.2 here,
    but feel free to create additional helper functions, should you need them
    '''
    ##########################################################################
    #TODO: Your code here



    pass
    ##########################################################################

def get_accuracy(c1, c2, c1_hat, c2_hat):
    '''
    Compute the accuracy of an assignment here!
    '''
    ##########################################################################
    #TODO: Your code here



    return
    ##########################################################################

def q1_3():
    '''
    You can probably just implement everything required for question 1.3 here,
    but feel free to create additional helper functions, should you need them
    '''
    ##########################################################################
    #TODO: Your code here



    pass
    ##########################################################################

if __name__ == "__main__":
    # Questions
    q1_1()
    #q1_2()
    #q1_3()
    print "Done with Question 1!\n"
