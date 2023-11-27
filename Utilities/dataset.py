import os
import json
import torch
import networkx as nx
from torch_geometric.data import InMemoryDataset, Data
from Utilities.custom_functions import from_networkx

class GraphDataset(InMemoryDataset):
    def __init__(self, root, ann_file: str, node_atts= ["x", "y", "z", "r", "topology"], edge_atts=['weight'], transform=None, pre_transform=None):
        self.node_atts = node_atts #["x", "y", "z", "t", "r", "topology", "side]
        self.edge_atts = edge_atts #weight/euclidean_distance
        super(GraphDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.ann_file = ann_file

    @property
    def raw_file_names(self):
        return ['graph_annotation.json']

    @property
    def processed_file_names(self):
        return ['processed.pt']

    def download(self):
        # You can implement downloading logic here if needed
        pass

    def process(self):
        with open(os.path.join(self.raw_dir, self.ann_file), 'r') as json_file:
            g_annotation = json.load(json_file)

        #Get graph categories dictionary (id: name)
        self.id_to_cat= {category["id"]: category["name"] for category in g_annotation["categories"]}
        self.cat_to_id= {category["name"]: category["id"] for category in g_annotation["categories"]}

        # Process graphs
        data_list = []
        for graph_data in g_annotation["graphs"]:
            graph_id = graph_data["id"]
            category_id = graph_data["category_id"]
            file_name = graph_data["file_name"]

            # Load the graph from the GML file
            graph = nx.read_gml(os.path.join(self.root, file_name))

            # Convert the NetworkX graph to PyTorch Geometric Data
            pyGeo_Data = from_networkx(graph, self.node_atts, self.edge_atts)
            # adding the ground truth at graph level to assign to each class the right/left label
            pyGeo_Data.__setattr__("y", category_id)

            # Add the graph to the list
            data_list.append(pyGeo_Data)

        # Save the processed data
        torch.save(self.collate(data_list), self.processed_paths[0])
