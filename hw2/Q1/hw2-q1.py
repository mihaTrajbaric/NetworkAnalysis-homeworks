import snap
import numpy as np

#FIn = snap.TFIn('hw2-q1.graph')
#G = snap.TUNGraph.Load(FIn)

G = snap.TUNGraph.Load(snap.TFIn("hw2-q1.graph"))

def neighbours(G,nId):
    '''

    :param: G graph
    :param: node id of certain node in graph
    :return: vect (list) of neighbor IDs
    '''
    node = G.GetNI(nId)


    nbrVect = []
    for Id in node.GetOutEdges():
        nbrVect.append(Id)
    return nbrVect


def featureVect(G, nId):
    '''
    :param nId: graph node Id
    :param G: graph (undirected)
    :return: feature vector of [deg,#egoInsideEdges,#egooutedges]
    '''
    node = G.GetNI(nId)
    degree = node.GetDeg()

    NIdV = snap.TIntV()
    for Id in node.GetOutEdges():
        NIdV.Add(Id)
    egoNet = snap.GetSubGraph(G, NIdV)
    egoinsideEdges = egoNet.GetEdges()
    egoOutEdges = 0
    for id in NIdV:
        nodeTemp = G.GetNI(id)
        for dstNiD in nodeTemp.GetOutEdges():
            if dstNiD not in NIdV and dstNiD != nId:
                egoOutEdges += 1

    return [degree,egoinsideEdges+degree,egoOutEdges]

def cosineSim(G,nId_1,nId_2,V):
    '''

    :param G: undirected graph
    :param nId_1: id of first node
    :param nId_2: id of second node
    :param V: dictionary of vectors
    :return: cosine similarity between the nodes
    '''
    if len(V) == 0:
        vect_1 = np.array(featureVect(G,nId_1))
        vect_2 = np.array(featureVect(G,nId_2))
    else:
        vect_1 = np.array(V[nId_1])
        vect_2 = np.array(V[nId_2])

    dot_product = np.dot(vect_1,vect_2)
    norm_1 = np.linalg.norm(vect_1)
    norm_2 = np.linalg.norm(vect_2)
    if norm_1 == 0 or norm_2 == 0:
        return 0
    return dot_product / (norm_1 * norm_2)




def featureVectTest():
    G1 = snap.TUNGraph.New()
    for i in range (1,9):
        G1.AddNode(i)
    G1.AddEdge(1, 2)
    G1.AddEdge(1, 3)
    G1.AddEdge(1, 4)

    G1.AddEdge(2, 3)
    G1.AddEdge(2, 4)

    G1.AddEdge(2, 5)
    G1.AddEdge(3, 6)
    G1.AddEdge(4, 7)
    G1.AddEdge(4, 8)

    vect = featureVect(1,G1)
    print "deg (3):",vect[0]
    print "inEdges (2):",vect[1]
    print "outEdges (4):",vect[2]

def recursive(G,V):
    """

    :param G undirected graph:
    :param V matrix of feature vector from i-1 th iteration:
    :return: matrix of i-th iteration
    """

    dictionary = {}
    if len(V) == 0:
        for node in G.Nodes():
            nIdtemp = node.GetId()
            nodeVect = featureVect(G, nIdtemp)
            dictionary[nIdtemp] = nodeVect
        return dictionary
    dim = len(V[G.GetRndNId()])
    for node in G.Nodes():
        nId = node.GetId()
        nbrList = neighbours(G,nId)
        deg = node.GetDeg()
        nbrMatrix = []
        for nbr in nbrList:
            nbrMatrix.append(V[nbr])
        mean =  np.mean(nbrMatrix,axis=0).tolist()
        sumVect = np.sum(nbrMatrix,axis=0).tolist()
        if np.isnan(mean).any():
            mean = []
            for i in range(dim):
                mean.append(0)
        if sumVect == 0:
            sumVect = []
            for i in range(dim):
                sumVect.append(0)
        dictionary[nId] = V[nId] + mean + sumVect
    return dictionary


def topN(G, nId, V, n = -1):
    '''

    :param n: how many to return (default -1 is return all)
    :param G: undirected graph
    :param V: dictionray of feature vectors
    :param nId: id of node
    :return: list of top five most similar nIds by cosineSim
    '''
    if len(V) == 0:
        V = recursive(G,V)

    sim = []
    for node in G.Nodes():
        nIdtemp = node.GetId()
        score = cosineSim(G, nId, nIdtemp,V)
        sim.append((nIdtemp, score))
    sim.sort(key=lambda x: x[1], reverse=True)
    if n == -1:
        return sim
    return sim[1:n+1]
def histogram(sim):
    """

    :param sim: sim scores for all nodes
    :return: none
    """
    import matplotlib.pyplot as plt
    input = np.array([x[1] for x in sim])
    plt.hist(input,bins = 100)
    plt.ylabel('#of nodes')
    plt.xlabel('score')
    plt.title("histogram of sim scores")
    plt.savefig('q1_3.png', format='png')
    plt.show()


def rnd_score(sim,V,lower, upper):
    """

    :param sim: sim scores of all nodes
    :param V: dict of vectors
    :param lower: lower boundary
    :param upper: upper boundary
    :return: some random vector
    """
    rndScore = (0,0)
    for score in sim:
        if score[1]>lower and score[1]<upper:
            rndScore = score
            break
    return [score[0],score[1],V[score[0]]]

def feature_vec_test(G,nId):

    #print "deg of nine", G.GetNI(9).GetDeg()
    #for n in neighbours(G, 9):
    #    print n, G.GetNI(9).GetDeg()
    node = G.GetNI(nId)
    degree = node.GetDeg()

    NIdV = snap.TIntV()
    for Id in node.GetOutEdges():
        NIdV.Add(Id)
    for i in NIdV:
        print i
    #print "vector sosedov:",NIdV
    egoNet = snap.GetSubGraph(G, NIdV)
    print "egonet:",egoNet.Dump()
    egoinsideEdges = egoNet.GetEdges()
    egoOutEdges = 0
    for id in NIdV:
        nodeTemp = G.GetNI(id)
        for dstNiD in nodeTemp.GetOutEdges():
            bool = dstNiD not in NIdV and dstNiD != nId
            if bool:
                egoOutEdges += 1

    return [degree, egoinsideEdges + degree, egoOutEdges]

if __name__ == '__main__':

    #print feature_vec_test(G, 9)

    print "1.1"
    print "feature vect for 9:", featureVect(G, 9)
    print "top 5 most similar nodes: "
    for element in topN(G, 9, {}, 5):
        print element
    print "_________________________"

    print "1.2"
    V = {}
    for i in range(2):
        V = recursive(G, V)
    print "feature vect 2nd iteration for 9:", V[9]
    V = recursive(G, V)
    print "feature vect 3nd iteration for 9:", V[9]
    print "top 5 most similar nodes: "
    for element in topN(G, 9, V, 5):
        print element
    print "_________________________"
    print "1.3"
    sim = topN(G,9,V)
    histogram(sim)
    print "sample node vectors"
    #spikes at scores
    spikes = [0.0,0.6,0.85, 0.9]
    for spike in spikes:
        print(rnd_score(sim,V,spike,spike+0.01))
    nodes = [1582,429,16,2]
    for nodeId in nodes:
        node = G.GetNI(nodeId)
        degree = node.GetDeg()

        NIdV = snap.TIntV()
        NIdV.Add(nodeId)
        for Id in node.GetOutEdges():
            NIdV.Add(Id)
        egoNet = snap.GetSubGraph(G, NIdV)
        egoinsideEdges = egoNet.GetEdges()
        print "nodes:",degree
        egoNet.Dump()




