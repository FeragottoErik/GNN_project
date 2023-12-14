import os
import json
import torch
import networkx as nx
import warnings
from torch_geometric.data import Dataset
from utils.custom_functions import from_networkx
import unittest
import random
from utils.augmentations import apply_augmentation_to_graph
import hcatnetwork

class ArteryGraphDataset(Dataset): #it is not an InMemoryDataset because it is not possible to load all the graphs in memory
    """ArteryGraphDataset is a custom dataset for artery tree graph level classification (i.e. left/right classification
    one reference label per sample and not for each node) that given a set of graphs in GML format (in root) and a JSON file 
    (ann_file) with COCO style annotations (i.e. a 'graphs' field where each item has a progressive ID, a file_name that specifies
    graph GML file location, coronary ostium node ID and the category_id which is either 1(left) or 2(right)) transforms
    the graphs into PyTorch Geometric Data objects thanks to from_networkx function enclosing the graph attributes
    in node (node_atts), edge (edge_atts) in a way compatible with pytorch geometric data sample representation.
    The graph-level label is assigned to the 'y' attribute of the Data object which
    accounts for the reference label. The ArteryGraphDataset class is a subclass of InMemoryDataset and inherits all its methods.
    enabling to seamlesly use it in the training pipeline.

    Args:
        root (str): The root directory of the dataset with 'raw' and 'processed' subfolders (processed subfolder will be
            created if not present to store the pytorch geometric info of the graphs and annotations).
        ann_file (str): The path to the JSON file with COCO style annotations (included in subfolder 'raw').
        node_atts (list): List of node attributes to include in the PyTorch Geometric Data object.
        edge_atts (list): List of edge attributes to include in the PyTorch Geometric Data object.
        transform (callable, optional): A function/transform that takes in a PyTorch Geometric Data object and returns a transformed version. Default is None.
        pre_transform (callable, optional): A function/transform that takes in a PyTorch Geometric Data object and returns a pre-processed version. Default is None.

    Returns:
        None

    """
    def __init__(self, root, ann_file: str, node_atts: list = ["x", "y", "z", "r", "topology"], edge_atts: list = ['weight'], augment: float = None, transform=None, pre_transform=None):
        self.node_atts = node_atts #["x", "y", "z", "t", "r", "topology", "side"]
        self.edge_atts = edge_atts #weight/euclidean_distance
        #nodes and edges attributes list must be assigned before calling the super constructor because it is used in the process method
        self.ann_file = ann_file
        self.VERBOSE = True
        self.augment = augment
        with open(os.path.join(root,'raw' , self.ann_file), 'r') as json_file:
            self.g_annotation = json.load(json_file)

        super(ArteryGraphDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return [self.ann_file] #the annotation file is the only raw file needed within the raw folder

    @property
    def processed_file_names(self):
        "if this file is present in the processed folder, the dataset is not processed again"
        try:
            with open(os.path.join(self.root, 'ref_data_list.txt'), 'r') as file:
                # Read the lines from the file and remove newline characters
                graphs_list = [line.strip() for line in file]
                return  graphs_list
        except: 
            return 'no_item' #pre_filter checkpoint from pytorch is always present when the dataset is processed and saved
        #even when there is no actual graph in the dataset, this only indicates the graph has been once processes and 
        #the 'processed' folder is not empty and present in the root folder along with the 'raw' folder

    def download(self):
        # no download is needed at the moment
        pass

    def process(self):
        with open(os.path.join(self.raw_dir, self.ann_file), 'r') as json_file:
            g_annotation = json.load(json_file)

        #Get graph categories dictionary (id: name)
        self.id_to_cat= {category["id"]: category["name"] for category in g_annotation["categories"]}
        self.cat_to_id= {category["name"]: category["id"] for category in g_annotation["categories"]}

        # Process graphs
        idx = 0
        for graph_data in g_annotation["graphs"]:
            if self.VERBOSE:
                print("Processing graph", idx, 'of', len(g_annotation["graphs"]))
            #graph_id = graph_data["id"]
            category_id = graph_data["category_id"]
            file_name = graph_data["file_name"]

            # Load the graph from the GML file
            graph = nx.read_gml(os.path.join(self.root, file_name))

            # Convert the NetworkX graph to PyTorch Geometric Data
            pyGeo_Data = from_networkx(graph, self.node_atts, self.edge_atts)

            #assert pyGeo_Data.x is not None and pyGeo_Data.edge_index is not None and pyGeo_Data.edge_attr is not None, "The graph is not correctly converted to PyTorch Geometric Data object"
            """The pyGeo_Data object is a PyTorch Geometric Data object with the following attributes:
            - pyGeo_Data.x: Node feature matrix with shape [num_nodes, num_node_features]
            - pyGeo_Data.edge_index: Graph connectivity in COO format with shape [2, num_edges] and type torch.long
            - pyGeo_Data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]"""
            # adding the ground truth at graph level to assign to each class the right/left label
            if pyGeo_Data.y is not None:
                warnings.warn(f"The graph-level label is already assigned to the 'y' attribute of the Data object.") #this happens 
                #because one of the node attributes is y but is not added to the ndoe features so is never deleted from pytorch Data keys
            pyGeo_Data.y = torch.tensor([category_id], dtype=torch.long) #pyGeo_Data.y: Graph-level label with shape [1] and type torch.long
            # Add the graph to the list
            #data_list.append(pyGeo_Data)
            torch.save(pyGeo_Data, os.path.join(self.processed_dir, f'data_{idx}.pt'))
            idx += 1

        # Save the processed data
        # data, slices = self.collate(data_list)
        # torch.save((data, slices), self.processed_paths[0])

    def _get_node_feats_by_graph(self, graph_idx: int):
        """Returns a list of node features for a given graph ID"""
        return self[graph_idx].x
    
    def _get_edge_feats_by_graph(self, graph_idx: int):
        """Returns a list of edge features for a given graph ID"""
        return self[graph_idx].edge_attr
    
    def _get_edge_index_by_graph(self, graph_idx: int):
        """Returns a list of edge indices for a given graph ID"""
        return self[graph_idx].edge_index

    def len(self):
        return len(self.processed_file_names)
    
    def get_raw_netwrorkx_graph(self, idx):
        graph_data = self.g_annotation["graphs"][idx]
        file_name = graph_data["file_name"]
        g = hcatnetwork.io.load_graph(file_path=os.path.join(self.root, file_name),
                                      output_type=hcatnetwork.graph.SimpleCenterlineGraph)
        return g

    def get(self, idx):
        if self.augment is None:
            data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
        else:
            #extract a random float number, if grater than self.augment, the graph is not augmented
            random_number = random.random()
            if random_number > self.augment:
                data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
            else:
                data = self._augment_data(idx)
        return data
 
    def _augment_data(self, idx):
        graph_data = self.g_annotation["graphs"][idx]
        category_id = graph_data["category_id"]
        file_name = graph_data["file_name"]

        # Load the graph from the GML file
        graph = nx.read_gml(os.path.join(self.root, file_name))
        augmented_graph = apply_augmentation_to_graph(graph, ['all'], prob=1)

        # Convert the NetworkX graph to PyTorch Geometric Data
        pyGeo_Data = from_networkx(augmented_graph, self.node_atts, self.edge_atts)

        if pyGeo_Data.y is not None:
                warnings.warn(f"The graph-level label is already assigned to the 'y' attribute of the Data object.") #this happens 
                #because one of the node attributes is y but is not added to the ndoe features so is never deleted from pytorch Data keys
        pyGeo_Data.y = torch.tensor([category_id], dtype=torch.long) #pyGeo_Data.y: Graph-level label with shape [1] and type torch.long
        # Add the graph to the list

        return pyGeo_Data



class TestArteryGraphDataset(unittest.TestCase):
    #TODO: correct the tests to match the new dataset structure
    def setUp(self):
        self.dataset = ArteryGraphDataset(root='/home/erikfer/GNN_project/DATA/SPLITTED_ARTERIES_DATA/', ann_file='graphs_annotation.json')
    
    def test_len(self):
        self.assertEqual(len(self.dataset), 96)
    
    def test_get(self):
        data = self.dataset[0]
        self.assertEqual(data.y, torch.tensor([1], dtype=torch.long))
        self.assertEqual(data.x.shape, torch.Size([data.num_nodes, 5]))
        self.assertEqual(data.edge_index.shape, torch.Size([2, data.num_edges]))
        self.assertEqual(data.edge_attr.shape, torch.Size([data.num_edges, 1]))

    def test_get_node_feats_by_graph(self):
        node_feats = self.dataset._get_node_feats_by_graph(0)
        self.assertEqual(node_feats.shape, torch.Size([node_feats.shape[0], 5]))
    
    def test_get_edge_feats_by_graph(self):
        edge_feats = self.dataset._get_edge_feats_by_graph(0)
        self.assertEqual(edge_feats.shape, torch.Size([edge_feats.shape[0], 1]))
    
    def test_get_edge_index_by_graph(self):
        edge_index = self.dataset._get_edge_index_by_graph(0)
        self.assertEqual(edge_index.shape, torch.Size([2, edge_index.shape[1]]))
    
    def test_get_raw_netwrorkx_graph(self):
        g = self.dataset.get_row_netwrorkx_graph(0)
        self.assertEqual(g.grap)


if __name__ == '__main__':
    unittest.main()
    dataset=ArteryGraphDataset(root='/home/erikfer/GNN_project/DATA/SPLITTED_ARTERIES_Normalized/', ann_file='graphs_annotation.json')