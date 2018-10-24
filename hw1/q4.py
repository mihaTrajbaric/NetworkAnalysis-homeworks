import snap
import matplotlib.pyplot as plt

G = snap.TUNGraph.New()
gene_disease = []
genes_list = []
disease_list = []
HDN = snap.TUNGraph.New()
disease_code_name = dict()
disease_id_code = dict()

def read():
    global gene_disease
    global genes_list
    global disease_list
    global disease_code_name
    global disease_id_code
    file = open("all_gene_disease_associations.tsv", "r")

    file.readline()
    gene_disease = []
    for line in file:
        a = line.split()
        disease_code = str(a[2])
        diseaseInt = int(disease_code[1:])
        if (disease_code not in disease_code_name):
            disease_code_name[disease_code] = str(a)
        geneInt = int(a[0])
        gene_disease.append((geneInt, diseaseInt))

    #print len(gene_disease)
    # sort by genes
    gene_disease.sort(key=lambda x: x[0])

    # print gene_disease[:100]
    counter = 0
    old_gene = -1
    new_gene_disease = []
    for edge in gene_disease:
        # ce imamo se vedno isti gen
        if (edge[0] != old_gene):
            counter += 1
            old_gene = edge[0]
        newPair = (counter, edge[1])
        new_gene_disease.append(newPair)
    # print new_gene_disease[:100]
    gene_disease = new_gene_disease
    # sort by disease
    gene_disease.sort(key=lambda x: x[1])
    # print(gene_disease[:100])
    # diseases IDs are greater then 20000
    counter = 20000
    old_disease = -1
    new_gene_disease = []
    for edge in gene_disease:
        # ce imamo se vedno isti gen
        if (edge[1] != old_disease):
            counter += 1
            old_disease = edge[1]
            #adds pair id real id to dict
            disease_id_code[counter] = old_disease
        newPair = (edge[0], counter)
        new_gene_disease.append(newPair)
    # print new_gene_disease[:100]

    gene_disease = new_gene_disease
    genes_list = list(set([x[0] for x in gene_disease]))
    disease_list = list(set([x[1] for x in gene_disease]))
    # make graph out of it

    for gene in genes_list:
        G.AddNode(gene)
    for disease in disease_list:
        G.AddNode(disease)
    for edge in gene_disease:
        G.AddEdge(edge[0], edge[1])


def q4_1():


    #G.Dump()
    print "q4_1"
    print "genes count:",len(genes_list)
    #genes count 17074
    print "diseases count:", len(disease_list)
    #disease count 20370
    print "total nodes:", G.GetNodes()
    #total nodes 37444
    print "total edges:", G.GetEdges()
    #total edges 561119
    print "last gene", genes_list[-1]
    #last gene 17074
    print "last disease", disease_list[-1]
    #last disease 32767

    nodes = G.BegEI()
    genDeg_vect = []
    genId_vect = []
    disDeg_vect = []
    for node in G.Nodes():
        if (node.GetId() < 20000):
            genDeg_vect.append(node.GetDeg())
            genId_vect.append(node.GetId())
        else:
            disDeg_vect.append(node.GetDeg())

    genDeg_vect.sort()
    disDeg_vect.sort()
    from itertools import groupby
    genDegPlot = [(key, len(list(group))) for key, group in groupby(genDeg_vect)]
    disDegPlot = [(key, len(list(group))) for key, group in groupby(disDeg_vect)]

    gen_x,gen_y = zip(*genDegPlot)
    dis_x,dis_y = zip(*disDegPlot)
    gen_y = [float(y)/len(genes_list) for y in gen_y]
    dis_y = [float(y)/len(disease_list) for y in dis_y]

    plt.loglog(gen_x, gen_y, 'o', c='blue', alpha=1, markeredgecolor='none', label = 'genes')

    plt.loglog(dis_x, dis_y,  'o', c='green', alpha=1, markeredgecolor='none', label = 'diseases')



    plt.xlabel('Node Degree (log)')
    plt.ylabel('Proportion of Nodes with a Given Degree (log)')
    plt.title('Degree Distribution of genes and diseases')
    plt.legend()
    plt.show()

    FOut = snap.TFOut('GDNetwork.graph')
    G.Save(FOut)
    FOut.Flush()

    import csv
    geneDegList = zip(genId_vect,genDeg_vect)
    geneDict = dict(geneDegList)

    import csv

    with open('geneDegrees.csv', 'wb') as f:
        w = csv.writer(f)
        w.writerows(geneDict.items())

    print "end of q4_1"
def q4_2_aux():
    FIn = snap.TFIn('HDN.graph')
    HDN = snap.TUNGraph.Load(FIn)
    edgesN = HDN.GetEdges()
    verticesN = HDN.GetNodes()
    print "nodes in HDN:",verticesN
    print "edged in HDN:",edgesN
    density = float(edgesN)/(verticesN*(verticesN-1)/2)
    print "density of the graph is",density
    CSum = 0.0
    #HDN.Dump()
    for i in range (1000):
        NId = HDN.GetRndNId()
        Ctemp = snap.GetNodeClustCf(HDN, NId)
        #print Ctemp
        CSum += Ctemp
    c = float(CSum)/1000
    print "average Clustering coeficient in HDN:",c

def q4_2():
    FIn = snap.TFIn('GDNetwork.graph')
    G = snap.TUNGraph.Load(FIn)
    import csv
    #id:degree
    geneDict =dict()
    with open('geneDegrees.csv', "r") as file:
        for line in file:
            list = line.split()[0].split(',')
            geneDict[int(list[0])] = int(list[1])

    #create HDN
    #traverse over genes, create full graph for every node
    #adding nodes
    for node in G.Nodes():
        if (node.GetId()<20000):
            continue
        HDN.AddNode(node.GetId())
    #gene disease boundary is 20000 (nodeId)
    #17047 is max gene id
    maxId = max(geneDict.keys())
    #maxCliques = []
    #counter = 0
    for i in range (maxId,0,-1):
        gene = G.GetNI(i)
        genDeg = gene.GetDeg()
        neighbours = []
        for k in range(genDeg):
            neighbours.append(gene.GetNbrNId(k))
        #if (counter<10):
        #    maxCliques.append(neighbours)
        #    counter += 1
        #add edges among nodes
        for j in range (len(neighbours)-1):
            for z in range (j+1,len(neighbours)):
                #add edge
                HDN.AddEdge(neighbours[j],neighbours[z])
        print i



    FOut = snap.TFOut('HDN.graph')
    G.Save(FOut)
    FOut.Flush()
    print "end of q4_2"

def q4_3():
    #the cliques arise because several diseases share same genes
    #the size of K max represent max degree among gene nodes
    #therefore calculating size of K max is simply calculating max degree

    FIn = snap.TFIn('GDNetwork.graph')
    G = snap.TUNGraph.Load(FIn)
    degList = []
    for node in G.Nodes():
        if (node.GetId() > 20000):
            break
        degList.append(node.GetDeg())
    maxDegree = max(degList)
    print "size of the max clique (Kmax) is",maxDegree


    print "end of q4_3"

def q4_4():
    FIn = snap.TFIn('HDN.graph')
    contracted = snap.TUNGraph.Load(FIn)

    F2In = snap.TFIn('GDNetwork.graph')
    GDN = snap.TUNGraph.Load(F2In)

    #find nodes with degrees more than 250 in GDN
    degrees = []
    #genes have ids lower than 20000
    it = GDN.BegNI()
    print "ZACETEK"
    while (it.GetId()<20000):
        degrees.append((it.GetId(),it.GetDeg()))
        it.Next()
    degrees.sort(key=lambda x:x[1],reverse=True)
    to_be_contracted = []
    #i will contracted all cliques larger than 250
    for tuple in degrees:
        if tuple[1]>250:
            to_be_contracted.append(tuple)
    #print len(to_be_contracted)
    to_be_contracted_sets = []
    for tuple in to_be_contracted:
        diseases_set = neigbours(GDN,tuple[0])
        to_be_contracted_sets.append(diseases_set)
    superNodes = set()
    for seti in to_be_contracted_sets:
        edgeSet = set()
        #first node is superNode, for every other node collect connections, than delete node and
        nodes = list(seti)
        superNode = nodes[0]
        for node in nodes:
            if node == superNode:
                continue
            edges = neigbours(contracted,node)
            for edg in edges:
                edgeSet.add(edg)
            if node not in superNodes:
                contracted.DelNode(node)
        #add connections to superNode, than put him to superNodes set
        superNodes.add(superNode)
        for edge in list(edgeSet):
            try:
                contracted.AddEdge(superNode,edge)
            except:
                pass

    edgesN = contracted.GetEdges()
    verticesN = contracted.GetNodes()
    print "nodes in contracted graph:", verticesN
    print "edged in contracted graph:", edgesN
    density = float(edgesN) / (verticesN * (verticesN - 1) / 2)
    print "density of the graph is", density
    CSum = 0.0
    for i in range(1000):
        NId = contracted.GetRndNId()
        Ctemp = snap.GetNodeClustCf(HDN, NId)

        CSum += Ctemp
    c = float(CSum) / 1000
    print "average Clustering coeficient in contracted:", c

    print "end of q4_4"
def neigbours(G,nodeId):
    node = G.GetNI(nodeId)
    nodeDeg = node.GetDeg()
    nbrSet = set()
    for i in range (nodeDeg):
        nbrid = node.GetNbrNId(i)
        nbrSet.add(nbrid)
    return nbrSet
def topFive(index,matrix):
    #retrurn top six matches as well as their codes and names
    matrix.sort(key=lambda x: x[index], reverse=True)
    for line in matrix[:6]:
        id = line[0]
        score = line[index]
        codeTemp = str(disease_id_code[id])
        temp = ""
        for i in range(7-len(codeTemp)):
            temp+="0"
        code = "C"+temp+codeTemp
        string = disease_code_name[code]
        print score,id,code,string
def q4_5():
    FIn = snap.TFIn('GDNetwork.graph')
    G = snap.TUNGraph.Load(FIn)

    #run together with read
    #disease_id_code
    #disease_code_name
    chron = "C0010346"
    leukemia = "C0023418"
    chron_forId = 10346
    leukemia_forId = 23418
    chrone_id = -1
    leukemia_id = -1
    #print chron,
    #print leukemia,disease_code_name[leukemia]
    for key, value in disease_id_code.items():
        if value == chron_forId:
            chrone_id = key
        elif value == leukemia_forId:
            leukemia_id = key
    #print chrone_id,chron, disease_code_name[chron]
    #print leukemia_id,leukemia,disease_code_name[leukemia]
    #disease are from 20000 north
    leukemiaSet = neigbours(G,leukemia_id)
    chroneSet = neigbours(G,chrone_id)

    maxId = G.GetMxNId()
    #vsaka vrstica je tuple (id,set, podobnost)
    matrix = []
    for i in range(20001,maxId):
        tempset = neigbours(G,i)
        union_leukemia = len(tempset.union(leukemiaSet))
        union_chron = len(tempset.union(chroneSet))
        intersection_leukemia = len(tempset.intersection(leukemiaSet))
        intersection_chron = len(tempset.intersection(chroneSet))
        ja_leukemia = float(intersection_leukemia)/union_leukemia
        ja_chron = float(intersection_chron)/union_chron
        temptuple = (i,intersection_chron,ja_chron,intersection_leukemia,ja_leukemia)
        matrix.append(temptuple)
    #chrone CN
    print "Chrohn Disease, CN"
    topFive(1,matrix)
    print "Chrohn Disease, JA"
    topFive(2,matrix)
    print "Leukemia, CN"
    topFive(3,matrix)
    print "Leukemia, JA"
    topFive(4,matrix)




    print "end of q4_5"





if __name__ == '__main__':
    read()
    q4_1()
    q4_2()
    q4_2_aux()
    q4_3()
    #q4_4()
    q4_5()
    print "end of q4"
