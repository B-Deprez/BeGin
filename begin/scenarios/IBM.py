import os
from sklearn.model_selection import train_test_split
import numpy as np

from data.IBM.label import create_identifiers, format_number, create_AML_labels, delete_specific_nodes

os.environ["DGLBACKEND"] = "pytorch"  # tell DGL what backend to use
import dgl
import torch
from dgl.data import DGLDataset

import pandas as pd

#from .utils.igraph_implementation import network_features

class IBMDataset_hom(DGLDataset):
    def __init__(self, dataset_name= "HI-Small"):
        self.dataset_name = dataset_name
        super().__init__(name="IBM_hom")
        self.num_classes = 8

    def define_ML_labels(self):
        path_trans="data/IBM/"+self.dataset_name+"_Trans.csv" 
        path_patterns="data/IBM/"+self.dataset_name+"_Patterns.txt"

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

        columns_money = ['Amount Received', 'Amount Paid']
        for col in columns_money: # make sure monetary amounts have two decimals
            transactions_df[col] = transactions_df[col].apply(lambda x: format_number(x))

        transactions_df['Is Laundering'] = transactions_df['Is Laundering'].astype(int)
        
        identifyer_list = create_identifiers(transactions_df)
        transactions_df["Identifyer"] = identifyer_list
        del identifyer_list

        pattern_columns = ["FAN-OUT", "FAN-IN", "GATHER-SCATTER", "SCATTER-GATHER", "CYCLE", "RANDOM", "BIPARTITE", "STACK"]
        df_patterns = create_AML_labels(path_patterns)

        # Merge the two dataframes
        transactions_df_extended = transactions_df.merge(df_patterns, on="Identifyer", how="left")
        transactions_df_extended = transactions_df_extended.fillna(0)

        # Vectorized check for "Is Laundering" being 1
        is_laundering = transactions_df_extended["Is Laundering"] == 1

        pattern_sum = transactions_df_extended[pattern_columns].max(axis=1) # To get a unique label for the patterns per node

        # Use numpy.where for a vectorized conditional operation
        transactions_df_extended["Not Classified"] = np.where((is_laundering) & (pattern_sum == 0), 1, 0)
            
        pattern_columns.append("Not Classified")

        return transactions_df_extended, pattern_columns
    
    def summarise_ML_labels(self,transactions_df_extended, pattern_columns):
        laundering_from = transactions_df_extended[["Account", "Is Laundering"]+pattern_columns].groupby("Account").mean()
        laundering_to = transactions_df_extended[["Account.1", "Is Laundering"]+pattern_columns].groupby("Account.1").mean()
        
        trans_from=transactions_df_extended[["Account", "Is Laundering"]+pattern_columns]
        trans_to=transactions_df_extended[["Account.1", "Is Laundering"]+pattern_columns]
        trans_to.columns = ["Account", "Is Laundering"]+pattern_columns
        laundering_combined = pd.concat([trans_from, trans_to]).groupby("Account").mean()

        return laundering_combined, laundering_from, laundering_to
    
    def create_labels(self, laundering_combined, pattern_columns, cutoff=0.2):
        i = 1 # To have different value for each pattern
        for col in pattern_columns:
            laundering_combined[col] = ((laundering_combined[col]>cutoff)*i).values
            i+=1
        laundering_combined["class"] = laundering_combined[pattern_columns].sum(axis=1)
  
        laundering_combined["class"] = laundering_combined["class"].apply(lambda x: x-1)

        laundering_combined.reset_index(inplace=True)
        laundering_combined=laundering_combined[["Account", "class"]]
        laundering_combined.columns = ["txId", "class"]
        
        return laundering_combined
    
    def create_node_data(self, transactions_df_extended):
        from_data = transactions_df_extended[["Account", "From Bank"]].drop_duplicates()
        from_data.columns = ["txId", "Bank"]
        to_data = transactions_df_extended[["Account.1", "To Bank"]].drop_duplicates()
        to_data.columns = ["txId", "Bank"]
        node_data = pd.concat([from_data, to_data], axis=0).drop_duplicates()

        # Convert bank names to integers
        node_data['Bank'], _ = pd.factorize(node_data['Bank'])

        return node_data
    
    def _masks(self, label_data, seed = 1997):
        mask_data = label_data.copy()
        mask_data['train_mask'] = False
        mask_data['val_mask'] = False
        mask_data['test_mask'] = False

        for step in mask_data["class"].unique():
            class_mask_data = mask_data[mask_data["class"] == step]

            train_index, test_index = train_test_split(class_mask_data.index, test_size=0.4, random_state=seed)
            val_index, test_index = train_test_split(test_index, test_size=0.5, random_state=seed)

            mask_data.loc[train_index, 'train_mask'] = True
            mask_data.loc[val_index, 'val_mask'] = True
            mask_data.loc[test_index, 'test_mask'] = True

        return mask_data['train_mask'], mask_data['val_mask'], mask_data['test_mask']

    def process(self):
        transactions_df, pattern_columns = self.define_ML_labels()
        laundering_combined, _, _ = self.summarise_ML_labels(transactions_df, pattern_columns)
        node_data = self.create_node_data(transactions_df)
        map_id = {j:i for i,j in enumerate(node_data["txId"])}
        node_data["txId"] = node_data["txId"].map(map_id)

        label_data = self.create_labels(laundering_combined, pattern_columns)
        label_data.txId = label_data.txId.map(map_id)

        edge_data = transactions_df[["Account", "Account.1", "Payment Format" , "Amount Received"]]
        edge_data.columns = ["txId1", "txId2", "Payment Format", "Amount Received"]
        edge_data.txId1 = edge_data.txId1.map(map_id)
        edge_data.txId2 = edge_data.txId2.map(map_id)

        edge_scr = torch.from_numpy(edge_data["txId1"].to_numpy())
        edge_dst = torch.from_numpy(edge_data["txId2"].to_numpy())
        edge_scr, edge_dst = torch.cat([edge_scr, edge_dst]), torch.cat([edge_dst, edge_scr]) # Make it bidirectional

        network_features = pd.read_csv("data/IBM/"+self.dataset_name+"_network.csv")
        node_data["degree"] = network_features["degree"]
        #node_data["betweenness"] = network_features["betweenness"]
        #node_data["closeness"] = network_features["closeness"]
        node_data["pagerank"] = network_features["pagerank"]

        node_features = torch.from_numpy(node_data.drop(columns=["txId"]).to_numpy())
        node_features = node_features.float()
        node_labels = torch.from_numpy(label_data["class"].to_numpy())

        self.graph = dgl.graph(
            (edge_scr, edge_dst), 
            num_nodes=node_data.shape[0]
            )
        
        self.graph.ndata["label"] = node_labels
        #self.graph.ndata["feat"] = node_features
        self.graph.ndata["feat"] =  torch.ones((node_data.shape[0], 1)).float()

        masks = self._masks(label_data)
        self.graph.ndata["train_mask"] = torch.from_numpy(masks[0].to_numpy())
        self.graph.ndata["val_mask"] = torch.from_numpy(masks[1].to_numpy())
        self.graph.ndata["test_mask"] = torch.from_numpy(masks[2].to_numpy())


    def __getitem__(self, i):
        return self.graph
    
    def __len__(self):
        return 1
    
class IBMDataset_heteronode(DGLDataset):
    def __init__(self, dataset_name= "HI-Small", separate_labels=False):
        self.dataset_name = dataset_name
        self.separate_labels = separate_labels
        super().__init__(name="IBM_het")

    def create_node_data(self, transactions_df):
        from_data = transactions_df[["Account", "From Bank"]].drop_duplicates()
        from_data.columns = ["txId", "Bank"]
        to_data = transactions_df[["Account.1", "To Bank"]].drop_duplicates()
        to_data.columns = ["txId", "Bank"]
        node_data = pd.concat([from_data, to_data], axis=0).drop_duplicates()

        # Convert bank names to integers
        node_data['Bank'], _ = pd.factorize(node_data['Bank'])

        return node_data
    
    def create_nodes(self, data):
        client_source = [] # Client who makes the transaction
        trans_target = [] # Transaction made by the client
        trans_source = [] # Transaction received by the client
        client_target = [] # Client who receives the transaction

        for i in range(len(data)):
            client_source.append(data.loc[i, "Account"])
            trans_target.append(i)
            trans_source.append(i)
            client_target.append(data.loc[i, "Account.1"])
        
        return client_source, trans_target, trans_source, client_target

    def create_graph(self, data):
        client_source, trans_target, trans_source, client_target = self.create_nodes(data)

        # Create the hererogeneous graph
        g = dgl.heterograph({
            ('client', 'sends', 'transaction'): (client_source, trans_target),
            ('transaction', 'received_by', 'client'): (trans_source, client_target), 
            ('client', 'receives', 'transaction'): (client_target, trans_source),
            ('transaction', 'sent_by', 'client'): (trans_target, client_source)
        })

        return g
    
    def create_labels(self, data, targets):
        labels = np.zeros(data.shape[0])
        for i in range(len(targets)):
            target_numeric = data[targets[i]]*(i+1)
            target_int = np.array(target_numeric.values, dtype=int)
            labels += target_int

        return labels

    def create_mask(self, data, train_size=0.6, val_size=0.2):
        # The masks are create to have a temporal split
        # Assumption that the data is sorted by timestamp in ascending order
        train_mask = np.zeros(data.shape[0])
        val_mask = np.zeros(data.shape[0])
        test_mask = np.zeros(data.shape[0])

        train_num = int(data.shape[0]*train_size)
        val_num = int(data.shape[0]*val_size)

        train_mask[:train_num] = 1
        val_mask[train_num:train_num+val_num] = 1
        test_mask[train_num+val_num:] = 1

        return train_mask, val_mask, test_mask

    def process(self):
        # Load data
        data = pd.read_csv('data/IBM/'+self.dataset_name+'_Trans_Patterns.csv')

        # Pre-processing
        # Convert the Timestamp column to datetime
        data['Timestamp'] = pd.to_datetime(data['Timestamp'])
        # Sort the data by timestamp, makes it easier to do temporal splits
        data = data.sort_values(by='Timestamp', ascending=True)
        data.reset_index(drop=True, inplace=True)

        # Extract day of the week, hour, and minute
        data['DayOfWeek'] = data['Timestamp'].dt.dayofweek
        data['Hour'] = data['Timestamp'].dt.hour
        data['Minute'] = data['Timestamp'].dt.minute

        node_data = self.create_node_data(data) # Create node data

        map_id = {j:i for i,j in enumerate(node_data["txId"])}
        data["Account"] = data["Account"].map(map_id)
        data["Account.1"] = data["Account.1"].map(map_id)
        node_data["txId"] = node_data["txId"].map(map_id)

        self.graph = self.create_graph(data)
        self.graph.nodes['client'].data['feat'] = torch.from_numpy(node_data.drop(columns=["txId"]).to_numpy()).float()
        
        columns_X = [
            'DayOfWeek', 
            'Hour', 
            'Minute', 
            'From Bank', 
            'To Bank', 
            'Amount Paid', 
            'Payment Currency', 
            'Receiving Currency',
            'Payment Format'
            ]
        targets = [
            #'Is Laundering',
            'FAN-OUT', 
            'FAN-IN', 
            'GATHER-SCATTER', 
            'SCATTER-GATHER', 
            'CYCLE',
            'RANDOM', 
            'BIPARTITE', 
            'STACK'
            ]
        
        # Change the data to one-hot encoding
        data = data[columns_X + ['Is Laundering'] + targets]
        data = pd.get_dummies(data)

        columns_X = list(data.columns.drop(targets)) # Update the columns_X to the new one-hot encoded columns

        self.graph.nodes['transaction'].data['feat'] = torch.from_numpy(data[columns_X].to_numpy()).float()

        if self.separate_labels:
            labels = self.create_labels(data, targets)
            self.graph.nodes['transaction'].data['label'] = torch.from_numpy(labels).float()
            self.num_classes = len(targets)+1 # Add one for the 0 label
        else:
            self.graph.nodes['transaction'].data['label'] = torch.from_numpy(data['Is Laundering'].to_numpy()).float()
            self.num_classes = 2

        masks = self.create_mask(data)

        self.graph.nodes['transaction'].data['train_mask'] = torch.from_numpy(masks[0]).bool()
        self.graph.nodes['transaction'].data['val_mask'] = torch.from_numpy(masks[1]).bool()
        self.graph.nodes['transaction'].data['test_mask'] = torch.from_numpy(masks[2]).bool()

    def __getitem__(self, i):
        return self.graph
    
    def __len__(self):
        return 1
    
class IBMDataset_link(DGLDataset):
    def __init__(self, dataset_name= "HI-Small", directed=True, separate_labels=False):
        self.dataset_name = dataset_name
        self.directed = directed
        self.separate_labels = separate_labels

        self.columns_X = ['DayOfWeek', 
            'Hour', 
            'Minute', 
            'From Bank', 
            'To Bank', 
            'Amount Paid', 
            'Payment Currency', 
            'Receiving Currency',
            'Payment Format'
            ]

        self.targets = [
            #'Is Laundering',
            'FAN-OUT', 
            'FAN-IN', 
            'GATHER-SCATTER', 
            'SCATTER-GATHER', 
            'CYCLE',
            'RANDOM', 
            'BIPARTITE', 
            'STACK'
            ]

        super().__init__(name="IBM_link")

    def process_data(self):
        # Load data
        data = pd.read_csv('data/IBM/'+self.dataset_name+'_Trans_Patterns.csv')

        # Pre-processing
        # Convert the Timestamp column to datetime
        data['Timestamp'] = pd.to_datetime(data['Timestamp'])
        # Sort the data by timestamp, makes it easier to do temporal splits
        data = data.sort_values(by='Timestamp', ascending=True)
        data.reset_index(drop=True, inplace=True)

        # Extract day of the week, hour, and minute
        data['DayOfWeek'] = data['Timestamp'].dt.dayofweek
        data['Hour'] = data['Timestamp'].dt.hour
        data['Minute'] = data['Timestamp'].dt.minute

        return data

    def create_node_data(self, transactions_df):
        from_data = transactions_df[["Account", "From Bank"]].drop_duplicates()
        from_data.columns = ["txId", "Bank"]
        to_data = transactions_df[["Account.1", "To Bank"]].drop_duplicates()
        to_data.columns = ["txId", "Bank"]
        node_data = pd.concat([from_data, to_data], axis=0).drop_duplicates()

        # Convert bank names to integers
        node_data['Bank'], _ = pd.factorize(node_data['Bank'])

        return node_data
    
    def create_labels(self, data, targets):
        labels = np.zeros(data.shape[0])
        for i in range(len(targets)):
            target_numeric = data[targets[i]]*(i+1)
            target_int = np.array(target_numeric.values, dtype=int)
            labels += target_int

        if not self.directed:
            labels = np.concatenate((labels, labels))

        return labels
    
    def process(self):
        data = self.process_data()
        node_data = self.create_node_data(data) # Create node data

        map_id = {j:i for i,j in enumerate(node_data["txId"])}
        node_data["txId"] = node_data["txId"].map(map_id)

        edge_data = data[["Account", "Account.1", "Payment Format" , "Amount Paid"]]
        edge_data.columns = ["txId1", "txId2", "Payment Format", "Amount Paid"]
        edge_data.txId1 = edge_data.txId1.map(map_id)
        edge_data.txId2 = edge_data.txId2.map(map_id)

        edge_features = torch.from_numpy(edge_data[["Amount Paid"]].to_numpy()).float()

        edge_scr = torch.from_numpy(edge_data["txId1"].to_numpy())
        edge_dst = torch.from_numpy(edge_data["txId2"].to_numpy())

        if not self.directed:
            edge_features = torch.cat([edge_features, edge_features])
            edge_scr, edge_dst = torch.cat([edge_scr, edge_dst]), torch.cat([edge_dst, edge_scr]) # Make it bidirectional

        self.graph = dgl.graph(
            (edge_scr, edge_dst), 
            num_nodes=node_data.shape[0]
        )

        # Change the data to one-hot encoding
        data = data[self.columns_X + ['Is Laundering'] + self.targets]
        data = pd.get_dummies(data)

        self.columns_X = list(data.columns.drop(self.targets)) # Update the columns_X to the new one-hot encoded columns

        self.graph.ndata["feat"] = torch.from_numpy(node_data.drop(columns=["txId"]).to_numpy()).float()
        self.graph.edata['feat'] = edge_features.float()

        if self.separate_labels:
            labels = self.create_labels(data, self.targets)
            self.graph.edata['label'] = torch.from_numpy(labels).float()
            self.num_classes = len(self.targets)+1 # Add one for the 0 label
        else:
            self.graph.edata['label'] = torch.from_numpy(data['Is Laundering'].to_numpy()).float()
            self.num_classes = 2

    def __getitem__(self, i):
        return self.graph
    
    def __len__(self):
        return 1