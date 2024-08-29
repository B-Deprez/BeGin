import os
from sklearn.model_selection import train_test_split
import numpy as np

os.environ["DGLBACKEND"] = "pytorch"  # tell DGL what backend to use
import dgl
import torch
from dgl.data import DGLDataset

import pandas as pd

from .utils.igraph_implementation import network_features

class IBMDataset(DGLDataset):
    def __init__(self, dataset_name= "HI-Small"):
        self.dataset_name = dataset_name
        super().__init__(name="IBM")
        self.num_classes = 8

    def create_identifiers(self, df):
        """
        Create a list of identifiers for each row in the dataframe.
        """
        # Convert all columns to string type
        df_str = df.astype(str)

        # Then use agg to join all column values into a single string for each row
        identifyer_list = df_str.agg(','.join, axis=1).tolist()

        return identifyer_list

    def format_number(self, number):
        formatted = str(number)
        if not('.' in formatted and len(formatted.split('.')[1]) >= 2):
            formatted = format(number, '.2f')
        return formatted 

    def create_AML_labels(self, path_patterns):
        transaction_list = []
        fanout_list = []
        fanin_list = []
        gather_scatter_list = []
        scatter_gather_list = []
        cycle_list = []
        random_list = []
        bipartite_list = []
        stack_list = []

        with open(path_patterns, "r") as f:
            attemptActive = False
            column = ""

            # Initialize all lists with zeros for simplification
            list_defaults = [0] * 8  # Assuming there are 8 lists as per the code snippet

            # Mapping of column names to their corresponding list index
            column_to_list_index = {
                "FAN-OUT": 0,
                "FAN-IN": 1,
                "GATHER-SCATTER": 2,
                "SCATTER-GATHER": 3,
                "CYCLE": 4,
                "RANDOM": 5,
                "BIPARTITE": 6,
                "STACK": 7
            }
            while True:
                line = f.readline()
                # Check if not at the end of the file
                if not line:
                    break

                # Add pattern to the corresponding transaction
                if line.startswith("BEGIN"): # Start of a pattern
                    attemptActive = True
                    column = line.split(" - ")[1].split(":")[0].strip()
                elif line.startswith("END"): # End of a pattern => reset all parameters + no update of columns
                    attemptActive = False
                    column = ""
                elif attemptActive:
                    identifyer = line.strip()
                    transaction_list.append(identifyer)
                    
                    # Reset all lists to default values
                    current_values = list_defaults.copy()
                    
                    if column in column_to_list_index:
                        # Update the relevant list based on the column name
                        current_values[column_to_list_index[column]] = 1
                        
                        # Unpack the updated values to each list
                        fanout_list.append(current_values[0])
                        fanin_list.append(current_values[1])
                        gather_scatter_list.append(current_values[2])
                        scatter_gather_list.append(current_values[3])
                        cycle_list.append(current_values[4])
                        random_list.append(current_values[5])
                        bipartite_list.append(current_values[6])
                        stack_list.append(current_values[7])

                    else:
                        raise ValueError("Unknown pattern type")
                    
        df_patterns = pd.DataFrame(
            {
                "Identifyer": transaction_list,
                "FAN-OUT": fanout_list,
                "FAN-IN": fanin_list,
                "GATHER-SCATTER": gather_scatter_list,
                "SCATTER-GATHER": scatter_gather_list,
                "CYCLE": cycle_list,
                "RANDOM": random_list,
                "BIPARTITE": bipartite_list,
                "STACK": stack_list
            }
        )

        return df_patterns

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
            transactions_df[col] = transactions_df[col].apply(lambda x: self.format_number(x))

        transactions_df['Is Laundering'] = transactions_df['Is Laundering'].astype(int)
        
        identifyer_list = self.create_identifiers(transactions_df)
        transactions_df["Identifyer"] = identifyer_list
        del identifyer_list

        pattern_columns = ["FAN-OUT", "FAN-IN", "GATHER-SCATTER", "SCATTER-GATHER", "CYCLE", "RANDOM", "BIPARTITE", "STACK"]
        df_patterns = self.create_AML_labels(path_patterns)

        # Merge the two dataframes
        transactions_df_extended = transactions_df.merge(df_patterns, on="Identifyer", how="left")
        transactions_df_extended = transactions_df_extended.fillna(0)

        # Vectorized check for "Is Laundering" being 1
        is_laundering = transactions_df_extended["Is Laundering"] == 1

        # Vectorized sum of specified columns
        pattern_sum = transactions_df_extended[pattern_columns].sum(axis=1)

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

        list_network_features = network_features(len(node_data), edge_scr)
        node_data["degree"] = list_network_features[0]
        node_data["betweenness"] = list_network_features[1]
        node_data["closeness"] = list_network_features[2]
        node_data["pagerank"] = list_network_features[3]

        node_features = torch.from_numpy(node_data.drop(columns=["txId"]).to_numpy())
        node_labels = torch.from_numpy(label_data["class"].to_numpy())

        self.graph = dgl.graph(
            (edge_scr, edge_dst), 
            num_nodes=node_data.shape[0]
            )
        
        self.graph.ndata["label"] = node_labels
        self.graph.ndata["feat"] = node_features


    def __getitem__(self, i):
        return self.graph
    
    def __len__(self):
        return 1




        

