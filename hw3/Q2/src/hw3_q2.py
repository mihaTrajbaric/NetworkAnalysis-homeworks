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

def distance(a,b):
    return np.linalg.norm(a-b)

def q2_3():
    G_snap.Dump()

    file =  open("Q2/emb/karateP100Q0.01.emd")
    dict = {}
    for line in file.readlines()[1:]:
        a =  line.split()
        dict[int(a[0])] = np.array([float(x) for x in a[1:]])
    distances = [(distance(dict[i],dict[33]),i) for i in range(1,35)]
    distances.sort(key=lambda x:x[0])
    print"top 5 most closest neighbors"
    for i in range(6):
        print distances[i]


def q2_4():
    file = open("Q2/emb/karateP1Q2.emd")
    dict = {}
    for line in file.readlines()[1:]:
        a = line.split()
        dict[int(a[0])] = np.array([float(x) for x in a[1:]])
    distances = [(distance(dict[i], dict[34]), i) for i in range(1, 35)]
    distances.sort(key=lambda x: x[0])
    print"top 5 most similar"
    for i in range(6):
        print distances[i]

if __name__ == '__main__':
    q2_3()
    q2_4()
