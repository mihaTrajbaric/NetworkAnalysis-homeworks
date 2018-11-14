###############################################################################
# CS 224W (Fall 2018) - HW2
# Starter code for Problem 3
###############################################################################
from __future__ import division
import snap
import matplotlib.pyplot as plt


# Setup
num_voters = 10000
decision_period = 10

class alter_states:
    def __init__(self):
        self.next_state = 'A'
    def __call__(self):
        to_return = self.next_state
        self.next_state = 'A' if self.next_state == 'B' else 'B'
        return to_return

def read_graphs(path1, path2):
    """
    :param - path1: path to edge list file for graph 1
    :param - path2: path to edge list file for graph 2

    return type: snap.PUNGraph, snap.PUNGraph
    return: Graph 1, Graph 2
    """
    ###########################################################################
    # DONE: Your code here!
    Graph1 = snap.LoadEdgeList(snap.PUNGraph,path1, 0, 1)
    Graph2 = snap.LoadEdgeList(snap.PUNGraph,path2, 0, 1)
    ###########################################################################
    return Graph1, Graph2


def initial_voting_state(Graph):
    """
    Function to initialize the voting preferences.

    :param - Graph: snap.PUNGraph object representing an undirected graph

    return type: Python dictionary
    return: Dictionary mapping node IDs to initial voter preference
            ('A', 'B', or 'U')

    Note: 'U' denotes undecided voting preference.

    Example: Some random key-value pairs of the dict are
             {0 : 'A', 24 : 'B', 118 : 'U'}.
    """
    voter_prefs = {}
    ###########################################################################
    # DONE: Your code here!
    for v in Graph.Nodes():
        id = v.GetId()
        last_digit = id % 10
        if last_digit in (0,1,2,3):
            voter_prefs[id] = 'A'
        elif last_digit in (4,5,6,7):
            voter_prefs[id] = 'B'
        else:
            voter_prefs[id] = 'U'
    ###########################################################################
    assert(len(voter_prefs) == num_voters)
    return voter_prefs

def friends_suport(v,conf):
    n_A = 0
    n_B = 0
    n_U = 0
    deg = v.GetDeg()
    for i in range(deg):
        nbr_id = v.GetNbrNId(i)
        state_nbr = conf[nbr_id]
        if state_nbr == 'A':
            n_A += 1
        elif state_nbr == 'B':
            n_B += 1
        else:
            n_U += 1
    assert n_A + n_B + n_U == deg
    return 'A' if n_A > n_B else ('B' if n_A < n_B else 'U')


def iterate_voting(Graph, init_conf, inflexible = []):
    """
    Function to perform the 10-day decision process.

    :param - inflexible: which nodes we cannot persuade regardles of digits
    :param - Graph: snap.PUNGraph object representing an undirected graph
    :param - init_conf: Dictionary object containing the initial voting
                        preferences (before any iteration of the decision
                        process)

    return type: Python dictionary
    return: Dictionary containing the voting preferences (mapping node IDs to
            'A','B' or 'U') after the decision process.

    Hint: Use global variables num_voters and decision_period to iterate.
    """
    curr_conf = init_conf.copy()
    alt_state = alter_states()
    ###########################################################################
    # DONE: Your code here!
    for _ in range(decision_period):
        for v in Graph.Nodes():
            id = v.GetId()
            if id%10 in (8, 9) and id%10 not in inflexible:
                config = friends_suport(v,curr_conf)
                if config == 'U':
                    curr_conf[id] = alt_state
                else:
                    curr_conf[id] = config


    ###########################################################################
    return curr_conf


def sim_election(Graph):
    """
    Function to simulate the election process, takes the Graph as input and
    gives the final voting preferences (dictionary) as output.
    """
    init_conf = initial_voting_state(Graph)
    conf = iterate_voting(Graph, init_conf)
    return conf


def winner(conf):
    """
    Function to get the winner of election process.
    :param - conf: Dictionary object mapping node ids to the voting preferences

    return type: char, int
    return: Return candidate ('A','B') followed by the number of votes by which
            the candidate wins.
            If there is a tie, return 'U', 0
    """
    ###########################################################################
    # DONE: Your code here!
    n_A = 0
    n_B = 0
    n_U = 0

    for key,value in conf.iteritems():
        if value == 'A':
            n_A += 1
        elif value == 'B':
            n_B += 1
        else:
            n_U += 1
    assert n_A + n_B + n_U == len(conf)
    if n_A > n_B:
        return 'A',n_A - n_B
    elif n_B > n_A:
        return 'B',n_B - n_A
    return 'U', 0
    ###########################################################################


def Q1():
    print ("\nQ1:")
    Gs = read_graphs('graph1.txt', 'graph2.txt')    # List of graphs

    # Simulate election process for both graphs to get final voting preference
    final_confs = [sim_election(G) for G in Gs]

    # Get the winner of the election, and the difference in votes for both
    # graphs
    res = [winner(conf) for conf in final_confs]

    for i in xrange(2):
        print "In graph %d, candidate %s wins by %d votes" % (
                i+1, res[i][0], res[i][1]
        )


def Q2sim(Graph, k):
    """
    Function to simulate the effect of advertising.
    :param - Graph: snap.PUNGraph object representing an undirected graph
             k: amount to be spent on advertising

    return type: int
    return: The number of votes by which A wins (or loses), i.e. (number of
            votes of A - number of votes of B)

    Hint: Feel free to use initial_voting_state and iterate_voting functions.
    """
    ###########################################################################
    # DONE: Your code here!
    assert k <= 9000
    voting_state = initial_voting_state(Graph)
    targeted = [x for x in range(3000,3000+int(k/100))]
    for i in targeted:
        voting_state[i] = 'A'
    conf = iterate_voting(Graph, voting_state,inflexible=targeted)
    w,margin = winner(conf)
    if w == 'B':
        margin *= -1
    return margin
    ###########################################################################


def find_min_k(diffs):
    """
    Function to return the minimum amount needed for A to win
    :param - diff: list of (k, diff), where diff is the value by which A wins
                   (or loses) i.e. (A-B), for that k.

    return type: int
    return: The minimum amount needed for A to win
    """
    ###########################################################################
    # DONE: Your code here!
    for k,diff in diffs:
        if diff > 0:
            return k
    ###########################################################################


def makePlot(res, title):
    """
    Function to plot the amount spent and the number of votes the candidate
    wins by
    :param - res: The list of 2 sublists for 2 graphs. Each sublist is a list
                  of (k, diff) pair, where k is the amount spent, and diff is
                  the difference in votes (A-B).
             title: The title of the plot
    """
    Ks = [[k for k, diff in sub] for sub in res]
    res = [[diff for k, diff in sub] for sub in res]
    ###########################################################################
    # title = ['graph1','graph2']
    for i in range(len(res)):
        plt.plot(Ks[i], res[i])

    ###########################################################################
    plt.plot(Ks[0], [0.0] * len(Ks[0]), ':', color='black')
    plt.xlabel('Amount spent ($)')
    plt.ylabel('#votes for A - #votes for B')
    plt.title(title)
    plt.legend(['graph1','graph2'])
    plt.show()


def Q2():
    print ("\nQ2:")
    # List of graphs
    Gs = read_graphs('graph1.txt', 'graph2.txt')

    # List of amount of $ spent
    Ks = [x * 1000 for x in range(1, 10)]

    # List of (List of diff in votes (A-B)) for both graphs
    res = [[(k, Q2sim(G, k)) for k in Ks] for G in Gs]

    # List of minimum amount needed for both graphs
    min_k = [find_min_k(diff) for diff in res]

    formatString = "On graph {}, the minimum amount you can spend to win is {}"
    for i in xrange(2):
        print formatString.format(i + 1, min_k[i])

    makePlot(res, 'TV Advertising')

def top_k(Graph,k):
    """
    Function to return top 9  nodes according to degree
    :param Graph: snap.PUNGraph object representing an undirected graph
    :return: list of node IDs
    """
    degrees = [(node.GetDeg(),node.GetId()) for node in Graph.Nodes()]
    degrees.sort(key=lambda x:x[0],reverse=True)
    # return [x[0] for x in degrees[:k]]
    return degrees[:k]


def Q3sim(Graph, k):
    """
    Function to simulate the effect of a dining event.
    :param - Graph: snap.PUNGraph object representing an undirected graph
    :param k: amount to be spent on the dining event

    return type: int
    return: The number of votes by which A wins (or loses), i.e. (number of
            votes of A - number of votes of B)

    Hint: Feel free to use initial_voting_state and iterate_voting functions.
    """
    ###########################################################################
    # DONE: Your code here!
    assert k <= 9000
    voting_state = initial_voting_state(Graph)
    targeted = [x[0] for x in top_k(Graph,k)]
    #targeted = [x for x in range(3000, 3000 + k / 100)]
    for i in targeted:
        voting_state[i] = 'A'
    conf = iterate_voting(Graph, voting_state, inflexible=targeted)
    w, margin = winner(conf)
    if w == 'B':
        margin *= -1
    return margin
    ###########################################################################


def Q3():
    print ("\nQ3:")
    # List of graphs
    Gs = read_graphs('graph1.txt', 'graph2.txt')

    # List of amount of $ spent
    Ks = [x * 1000 for x in range(0, 10)]

    # List of (List of diff in votes (A-B)) for both graphs
    res = [[(k, Q3sim(G, k)) for k in Ks] for G in Gs]
    # List of minimum amount needed for both graphs
    min_k = [find_min_k(diff) for diff in res]

    formatString = "On graph {}, the minimum amount you can spend to win is {}"
    for i in xrange(2):
        print formatString.format(i + 1, min_k[i])

    makePlot(res, 'Wining and Dining')


def getDataPointsToPlot(Graph):
    """
    :param - Graph: snap.PUNGraph object representing an undirected graph

    return values:
    X: list of degrees
    Y: list of frequencies: Y[i] = fraction of nodes with degree X[i]
    """
    ############################################################################
    X, Y = [], []
    DegToCntV = snap.TIntPrV()
    snap.GetDegCnt(Graph, DegToCntV)
    for p in DegToCntV:
        X.append(p.GetVal1())
        Y.append(p.GetVal2())
    ############################################################################
    return X, Y

def Q4():
    """
    Function to plot the distributions of two given graphs on a log-log scale.
    """
    print ("\nQ4:")
    ###########################################################################
    # DONE: Your code here!

    Gs = read_graphs('graph1.txt', 'graph2.txt')
    for G in Gs:
        X, Y = getDataPointsToPlot(G)
        plt.loglog(X,Y)
    plt.xlabel('degree')
    plt.ylabel('count')
    plt.title("Degree distribution of two graphs")
    plt.legend(['graph1', 'graph2'])
    plt.show()
    ###########################################################################
def debug():
    Gs = read_graphs('graph1.txt', 'graph2.txt')
    k = 20
    for g in Gs:
        for node in top_k(g,k):
            print "deg:{} id:{}".format(node[0], node[1])
        print "__________________"

def main():
    Q1()
    Q2()
    Q3()
    Q4()

    # debug()

if __name__ == "__main__":
    main()
