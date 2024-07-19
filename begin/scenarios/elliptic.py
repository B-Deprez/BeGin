import os
from sklearn.model_selection import train_test_split

os.environ["DGLBACKEND"] = "pytorch"  # tell DGL what backend to use
import dgl
import torch
from dgl.data import DGLDataset

import pandas as pd

class EllipticDataset(DGLDataset):
    def __init__(self):
        super().__init__(name="elliptic")
        self.num_classes = 2

    def process(self):
        node_data = pd.read_csv("data/elliptic/elliptic_txs_features.csv", header=None)
        label_data = pd.read_csv("data/elliptic/elliptic_txs_classes.csv")
        edge_data = pd.read_csv("data/elliptic/elliptic_txs_edgelist.csv")

        map_id = {j:i for i,j in enumerate(node_data[0])}
        map_label = {'unknown': 2, '1': 1, '2': 0}
        label_data['class'] = label_data['class'].map(map_label)
        label_data['class'] = torch.from_numpy(label_data['class'].values)

        node_data[0] = node_data[0].map(map_id)
        label_data.txId = label_data.txId.map(map_id)   
        edge_data.txId1 = edge_data.txId1.map(map_id)
        edge_data.txId2 = edge_data.txId2.map(map_id)

        node_features = torch.from_numpy(node_data.drop(columns=[0, 1]).to_numpy())
        node_features = node_features.float()
        node_labels = torch.from_numpy(
            label_data["class"].to_numpy()
            )
        
        node_data[1] = node_data[1]//7

        time_stamp = torch.from_numpy(
            node_data[1].to_numpy()
            )

        edge_src = torch.from_numpy(edge_data["txId1"].to_numpy())
        edge_dst = torch.from_numpy(edge_data["txId2"].to_numpy())
        edge_src, edge_dst = torch.cat([edge_src, edge_dst]), torch.cat([edge_dst, edge_src])

        self.graph = dgl.graph(
            (edge_src, edge_dst), 
            num_nodes=node_data.shape[0]
            )
        self.graph.ndata["label"] = node_labels
        self.graph.ndata["feat"] = node_features
        self.graph.ndata["time"] = time_stamp
        self.labels = node_labels

        masks = self._masks(node_data, label_data)
        
        self.graph.ndata["train_mask"] = torch.from_numpy(masks[0].to_numpy())
        self.graph.ndata["val_mask"] = torch.from_numpy(masks[1].to_numpy())
        self.graph.ndata["test_mask"] = torch.from_numpy(masks[2].to_numpy())

    def __getitem__(self, i):
        return self.graph
    
    def __len__(self):
        return 1
    
    def _masks(self, node_data, label_data, seed = 1997):
        mask_data = node_data[[0, 1]].copy()
        mask_data['train_mask'] = False
        mask_data['val_mask'] = False
        mask_data['test_mask'] = False

        for step in mask_data[1].unique():
            step_mask_data = mask_data[mask_data[1] == step]
            step_mask_data = step_mask_data.merge(label_data, left_on=0, right_on='txId')
            step_mask_data = step_mask_data.set_index(0, drop=False)
            step_mask_data.index.name = None
            step_mask_data = step_mask_data[step_mask_data['class'] != 2]

            num_obs = step_mask_data.shape[0]
            num_train = int(num_obs * 0.6)
            num_val = int(num_obs * 0.2)

            try:
                train_data, temp_data = train_test_split(step_mask_data, train_size=num_train, stratify=step_mask_data['class'], random_state=seed)
                val_data, test_data = train_test_split(temp_data, train_size=num_val, stratify=temp_data['class'], random_state=seed)
                train_idx = train_data.index
                val_idx = val_data.index
                test_idx = test_data.index
                mask_data.loc[train_idx, 'train_mask'] = True
                mask_data.loc[val_idx, 'val_mask'] = True
                mask_data.loc[test_idx, 'test_mask'] = True

            except: # In case step has too few labels to do the split
                train_idx = step_mask_data.sample(n=num_train, random_state=seed).index
                step_mask_data.loc[train_idx, 'train_mask'] = True
                mask_data.loc[train_idx, 'train_mask'] = True
                val_idx = step_mask_data[~step_mask_data['train_mask']].sample(n=num_val, random_state=seed).index
                step_mask_data.loc[val_idx, 'val_mask'] = True
                mask_data.loc[val_idx, 'val_mask'] = True
                test_idx = step_mask_data[~step_mask_data['train_mask'] & ~step_mask_data['val_mask']].index
                mask_data.loc[test_idx, 'test_mask'] = True

        mask_data.sort_index(inplace=True)
        return mask_data['train_mask'], mask_data['val_mask'], mask_data['test_mask']