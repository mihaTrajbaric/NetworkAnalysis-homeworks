from __future__ import division
import snap
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


G_snap = snap.LoadEdgeList(snap.PUNGraph, "C:\\Users\\Miha\\PycharmProjects\\NetworkAnalysis\\hw3\\Q2\\graph\\karate.edgelist", 0, 1)
G = nx.read_edgelist(path="C:\\Users\\Miha\\PycharmProjects\\NetworkAnalysis\\hw3\\Q2\\graph\\karate.edgelist")
def plot():
    plt.subplot(121)
    nx.draw_spring(G, with_labels=True, font_weight='bold')
    plt.show()

def distance(u,v):
    # dist = 1.0 - np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    dist = np.linalg.norm(u-v)
    return dist

def q2_5():
    # G_snap.Dump()

    file =  open("Q2/emb/karateP100Q0.01.emd")
    dict = {}
    for line in file.readlines()[1:]:
        a =  line.split()
        dict[int(a[0])] = np.array([float(x) for x in a[1:]])
    distances = [(distance(dict[i],dict[33]),i) for i in range(1,35)]
    distances.sort(key=lambda x:x[0])
    print"top 5 most closest neighbors to 33"
    for i in range(6):
        print distances[i]

def q2_3():
    file = open("Q2/emb/karateP1Q0.01.emd")
    dict = {}
    for line in file.readlines()[1:]:
        a = line.split()
        dict[int(a[0])] = np.array([float(x) for x in a[1:]])
    distances = [(distance(dict[i], dict[33]), i) for i in range(1, 35)]
    distances.sort(key=lambda x: x[0])
    print"top 5 most closest neighbors to 33"
    for i in range(6):
        print distances[i]


def q2_4():
    file = open("Q2/emb/karateP0.5Q2.emd")
    dict = {}
    for line in file.readlines()[1:]:
        a = line.split()
        dict[int(a[0])] = np.array([float(x) for x in a[1:]])
    distances = [(distance(dict[i], dict[34]), i) for i in range(1, 35)]
    distances.sort(key=lambda x: x[0])
    print"structural similarity to 34 by DFS (P 0.01 Q 1"
    for i in range(6):
        print distances[i]
def print_custom():
    nodes = []
    for id in range(1,G_snap.GetNodes()+1):
        v = G_snap.GetNI(id)
        nbr_list = []
        for i in range(v.GetDeg()):
            nbr_list.append(v.GetNbrNId(i))
        nodes.append(( id,v.GetDeg(),nbr_list))
    nodes.sort(key=lambda x:x[1],reverse = True)
    for a in nodes:
        print a
if __name__ == '__main__':
    # q2_3()
    q2_4()
    # q2_5()
    print_custom()
