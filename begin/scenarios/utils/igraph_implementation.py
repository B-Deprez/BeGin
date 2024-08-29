import igraph as ig

def create_igraph(num_nodes, edges):
    g = ig.Graph(n=num_nodes, directed=False)
    g.add_edges(edges)
    return g

def network_features(num, edges):
    g = create_igraph(num, edges)

    degree = g.degree() #Degree
    betw = g.betweenness() #Betweenness
    cls = g.closeness() #Closeness
    pgrnk = g.pagerank() #PageRank

    return [degree, betw, cls, pgrnk]