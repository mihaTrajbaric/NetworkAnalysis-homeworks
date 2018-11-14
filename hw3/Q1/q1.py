from __future__ import division
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
    # DONE: Your code here
    n_nodes = Graph.GetNodes()
    A = np.zeros((n_nodes,n_nodes))
    for e in Graph.Edges():
        id1 = e.GetSrcNId()
        id2 = e.GetDstNId()
        A[id1, id2] = 1
        A[id2, id1] = 1
    return A




    ##########################################################################

def get_sparse_degree_matrix(Graph):
    '''
    This function might be useful for you to build the degree matrix of a
    given graph and return it as a numpy array
    '''
    ##########################################################################
    #DONE: Your code here
    n_nodes = Graph.GetNodes()
    D = np.zeros((n_nodes, n_nodes))
    for v in Graph.Nodes():
        id = v.GetId()
        D[id][id] = v.GetDeg()
    return D



    ##########################################################################

def normalized_cut_minimization(Graph):
    '''
    Implement the normalized cut minimizaton algorithm we derived in the last
    homework here
    '''
    A = get_adjacency_matrix(Graph)
    D = get_sparse_degree_matrix(Graph)
    ##########################################################################
    L = np.subtract(D,A)
    D_potenca = np.diag(1 / np.sqrt(np.diag(D)))
    L_normalized = np.matmul(np.matmul(D_potenca,L),D_potenca)
    # print A
    values, vects = np.linalg.eigh(L_normalized)
    # print values

    fidler = vects[:,1]
    # print fidler
    """
    plt.plot(sorted(fidler))
    plt.plot([0 for _ in fidler])
    plt.plot(12,0,'o')
    plt.show()
    """
    # a = sum([1 if x < 0 else 0 for x in fidler])
    # print a,len(fidler)-a
    cluster1 = set()
    cluster2 = set()
    i = 0
    clustering = []
    for v in Graph.Nodes():
        id = v.GetId()
        clustering.append((id,np.sign(fidler[i])))
        # print id, np.sign(fidler[i])
        if fidler[id] < 0:
            cluster1.add(id)
        else:
            cluster2.add(id)
        i += 1
    # print len(cluster1),len(cluster2)
    # print cluster1,cluster2
    """
    clustering.sort(key=lambda x:x[1])
    for pair in clustering:
        print pair
    """
    return cluster1,cluster2
    ##########################################################################

def modularity(Graph, c1, c2):
    '''
    This function might be useful to compute the modularity of a given cut
    defined by two sets c1 and c2. We would normally require sets c1 and c2
    to be disjoint and to include all nodes in Graph
    '''
    ##########################################################################
    #DONE: Your code here
    m = Graph.GetEdges()
    #vol S and S'

    c1_list = list(c1)
    c2_list = list(c2)

    vol_c1 = 0
    for id in c1_list:
        node = Graph.GetNI(id)
        vol_c1 += node.GetDeg()

    vol_c2 = 0
    for id in c2_list:
        node = Graph.GetNI(id)
        vol_c2 += node.GetDeg()

    #cut S
    cut_s = 0
    for v1 in c1_list:
        for v2 in c2_list:
            povezava = 1 if Graph.IsEdge(v1, v2) else 0
            cut_s += povezava
    mod = (1 / (2*m)) * (-2*cut_s + (1/m)* vol_c1 * vol_c2)
    return mod


    ##########################################################################

def q1_1():
    '''
    Main q1_1 workflow. All of the work can be done in gen_config_model_rewire
    but you may modify this function as needed.
    '''
    ##########################################################################
    #DONE: Your code here
    name_list = ["sample","normal","rewired"]
    for name in name_list:
        Graph = load_graph(name)
        c1,c2 = normalized_cut_minimization(Graph)
        # print len(c1),len(c2)
        Q = modularity(Graph,c1,c2)
        print "------------------"
        print "Graph {}".format(name)
        print "len(c1): {}, len(c2): {}, modularity: {}".format(len(c1),len(c2),Q)
        print "------------------"
        # print Q
        # Graph.Dump()
    ###########
    # ###############################################################

def SSBM(n,pi,a,b,verbose = False):
    '''

    '''
    ##########################################################################
    #DONE: Your code here
    G = snap.TUNGraph.New()
    # add n nodes
    for i in range(n):
        G.AddNode(i)
    # assign communities
    assert sum(pi) == 1
    communities = {}
    for i in range(len(pi)):

        first = int(sum(pi[:i])*n)

        next_ = int(first + pi[i] * n)
        for j in range(first,next_):
            communities[j] = i

        # print "first: {} next: {}".format(first,next_)
    """
    for key, value in communities.iteritems():
        print "id {} com {}".format(key,value)
    """
    import random
    for i in range(n):
        for j in range(i+1,n):
            com_i = communities[i]
            com_j = communities[j]
            prob = random.random()
            desired_prob = a if com_i == com_j else b
            if prob <= desired_prob:
                G.AddEdge(i, j)
    adj = get_adjacency_matrix(G)
    # print adj
    if verbose:
        plt.imshow(adj, cmap='binary')
        plt.title("n: {} pi: {} in: {} out: {}".format(n,pi,a,b))
        plt.show()
    return G

    ##########################################################################

def q1_2():
    '''
    You can probably just implement everything required for question 1.2 here,
    but feel free to create additional helper functions, should you need them
    '''
    ##########################################################################
    #DONE: Your code here
    SSBM(10, [0.5, 0.5], 0.9, 0.3,verbose=True)
    SSBM(100, [0.1, 0.2, 0.3, 0.4], 0.6, 0.15,verbose=True)
    SSBM(1000, [0.25, 0.25, 0.25, 0.25], 0.2, 0.6,verbose=True)
    SSBM(1000, [0.25, 0.35, 0.1, 0.3], 0.7, 0.5,verbose=True)
    ##########################################################################

def get_accuracy(c1, c2, c1_hat, c2_hat):
    '''
    Compute the accuracy of an assignment here!
    '''
    ##########################################################################
    #DONE: Your code here
    intersect_1 = len(c1 & c1_hat)
    intersect_2 = len(c2 & c2_hat)

    acc = 2 * ((intersect_1 + intersect_2) / (len(c1) + len(c2)) - 0.5)
    return acc
    ##########################################################################

def accuracy(n,pi,prob_in,prob_out,verbose = False):
    G = SSBM(n, pi, prob_in, prob_out, verbose)
    c1, c2 = normalized_cut_minimization(G)
    c1_hat = set([x for x in range(int(n * pi[0]))])
    c2_hat = set([x for x in range(int(n * pi[0]), n)])
    assert len(c1_hat & c2_hat) == 0
    acc = max(get_accuracy(c1, c2, c1_hat, c2_hat), get_accuracy(c1, c2, c2_hat, c1_hat))
    return acc


def exact_recovery(a,b):
    left = (a + b) / 2
    right = 1 + np.sqrt(a * b)
    return left > right


def weak_recovery(a,b):
    left = ((a - b) ** 2) / (2 * (a + b))
    right = 1
    return left > right

def boundary (a1,a2,faktor,const,n):
    X = [x/faktor for x in range(a1,a2)]
    y = const
    defult_exact = exact_recovery(X[0]*n,y*n)
    defult_weak = weak_recovery(X[0]*n,y*n)
    exact_i = []
    weak_i = []
    for i in range(len(X)):
        temp_exact = exact_recovery(X[i]*n,y*n)
        temp_weak = weak_recovery(X[i]*n,y*n)
        #print"{}: a {} b {} exact {} weak {}".format(i,X[i],y,temp_exact,temp_weak)
        if temp_exact != defult_exact:
            exact_i.append(i)
        if temp_weak != defult_weak:
            weak_i.append(i)
    # print 'exact',exact_i,"weak",weak_i
    # TODO dodaj parameter first or last (za p_out je prva, za p_in zadnja?)
    return X[exact_i[0]],X[weak_i[0]]

def test():
    """
    exact,weak = boundary(1,500,500,0.3,1000)
    print "-----------P_IN----------"
    print "exact: {} weak {}".format(exact,weak)
    """
    exact, weak = boundary(1, 500, 500, 0.7, 1000)
    print "-----------P_OUT----------"
    print "exact: {} weak {}".format(exact, weak)

def q1_3():
    '''
    You can probably just implement everything required for question 1.3 here,
    but feel free to create additional helper functions, should you need them
    '''
    ##########################################################################
    # DONE: Your code here

    n = 1000
    pi = [0.5,0.5]
    prob_in = 0.55
    prob_out = 0.5

    print accuracy(n,pi,prob_in,prob_out)
    """
    prob_out = 0.3
    acc_list = []

    for i in range(150,200):
        prob_in = i/500
        acc = accuracy(n,pi,prob_in,prob_out)
        print prob_in,acc
        acc_list.append((prob_in,acc))
    print acc_list
    
    plt.plot([x[0] for x in acc_list],[x[1] for x in acc_list])
    exact,weak = boundary(1,500,500,0.3,1000)
    plt.axvline(x=exact)
    plt.axvline(x=weak)
    plt.title("accuracy aggainst P_in")
    plt.xlabel("P_in")
    plt.ylabel("accuracy")
    plt.show()

    prob_in = 0.7
    acc_list = []
    for i in range(300, 350):
        prob_out = i/500
        acc = accuracy(n, pi, prob_in, prob_out)
        print prob_out, acc
        acc_list.append((prob_out, acc))
    print acc_list

    plt.plot([x[0] for x in acc_list], [x[1] for x in acc_list])
    exact, weak = boundary(1, 500, 500, 0.7, 1000)
    plt.axvline(x=exact)
    plt.axvline(x=weak)
    plt.title("accuracy aggainst P_out")
    plt.xlabel("P_out")
    plt.ylabel("accuracy")
    plt.show()
    """
    # prob_in sweep
    # prob_out in sweep




    pass
    ##########################################################################

if __name__ == "__main__":
    # Questions

    # q1_1()
    # q1_2()
    q1_3()
    # test()
    print "Done with Question 1!\n"
