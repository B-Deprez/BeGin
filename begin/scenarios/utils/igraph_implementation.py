import igraph as ig
import torch
import pandas as pd
import numpy as np

def create_igraph(num_nodes, edges):
    g = ig.Graph(n=num_nodes, directed=False)
    g.add_edges(zip(*edges))
    return g

def network_features(num, edges):
    g = create_igraph(num, edges)

    degree = g.degree() #Degree
    print("Degree done")
    #betw = g.betweenness() #Betweenness
    betw = 0
    print("Betweenness done")
    #cls = g.closeness() #Closeness
    cls = 0
    print("Closeness done")
    pgrnk = g.pagerank() #PageRank
    print("PageRank done")

    return [degree, betw, cls, pgrnk]

def create_node_data(transactions_df_extended):
    from_data = transactions_df_extended[["Account", "From Bank"]].drop_duplicates()
    from_data.columns = ["txId", "Bank"]
    to_data = transactions_df_extended[["Account.1", "To Bank"]].drop_duplicates()
    to_data.columns = ["txId", "Bank"]
    node_data = pd.concat([from_data, to_data], axis=0).drop_duplicates()

    # Convert bank names to integers
    node_data['Bank'], _ = pd.factorize(node_data['Bank'])

    return node_data

def load_transactions_df(dataset_name):
    path_trans="data/IBM/"+dataset_name+"_Trans.csv" 

    dtype_dict = {
            "From Bank": str,
            "To Bank": str,
            "Account": str,
            "Account.1": str
        }

    transactions_df = pd.read_csv(path_trans, dtype=dtype_dict)

    nodes_to_delete = [
        "81211BC00",
        "8135B8250",
        "80FA55EF0",
        "80A7FD400",
        "81211BA20",
        "8135B8200",
        "80FA56340",
        "80A7FDE00"
        ] # These accounts pose some problems, where the account is linked to two different banks
    
    transactions_df = transactions_df[~transactions_df["Account"].isin(nodes_to_delete)]
    transactions_df = transactions_df[~transactions_df["Account.1"].isin(nodes_to_delete)]

    return transactions_df

if __name__ == "__main__":
    dataset_name = "HI-Small"
    print("Loading transactions data")
    transactions_df = load_transactions_df(dataset_name)
    print("Creating network")
    node_data = create_node_data(transactions_df)
    map_id = {j:i for i,j in enumerate(node_data["txId"])}
    node_data["txId"] = node_data["txId"].map(map_id)

    edge_data = transactions_df[["Account", "Account.1", "Payment Format" , "Amount Received"]]
    edge_data.columns = ["txId1", "txId2", "Payment Format", "Amount Received"]
    edge_data.txId1 = edge_data.txId1.map(map_id)
    edge_data.txId2 = edge_data.txId2.map(map_id)

    edge_scr = torch.from_numpy(edge_data["txId1"].to_numpy())
    edge_dst = torch.from_numpy(edge_data["txId2"].to_numpy())
    edge_scr, edge_dst = torch.cat([edge_scr, edge_dst]), torch.cat([edge_dst, edge_scr]) # Make it bidirectional

    print("Computing network features")
    list_network_features = network_features(len(node_data), [edge_scr,edge_dst])
    node_data["degree"] = list_network_features[0]
    node_data["betweenness"] = list_network_features[1]
    node_data["closeness"] = list_network_features[2]
    node_data["pagerank"] = list_network_features[3]

    network_data = node_data[["txId", "degree", "betweenness", "closeness", "pagerank"]]
    print("Saving network data")
    network_data.to_csv("data/IBM/"+dataset_name+"_network.csv", index=False)