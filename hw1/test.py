import snap

a = snap.TUNGraph.New()
for i in range (5):
	a.AddNode(i)
a.AddEdge( 0, 1)
a.AddEdge( 0, 2)
a.AddEdge( 0, 3)
a.AddEdge( 0, 4)
a.AddEdge( 1, 2)
a.AddEdge( 2, 3)
a.AddEdge( 3, 4)
a.Dump()
print type(a.GetNI(0).GetDeg())
    