import networkx as nx
from hcatnetwork.graph import SimpleCenterlineGraph
import hcatnetwork
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import json
import numpy as np

import scipy.sparse
import torch
from torch import Tensor
from torch.utils.dlpack import from_dlpack, to_dlpack
import os
import torch_geometric
from torch_geometric.utils.num_nodes import maybe_num_nodes
from hcatnetwork.node import ArteryNodeTopology
from torch_geometric.data import Data

def get_processed_graphs_names_and_write_reference_txt(folder_path='/home/erikfer/GNN_project/DATA/SPLITTED_ARTERIES_Normalized/processed',\
                                                        root='/home/erikfer/GNN_project/DATA/SPLITTED_ARTERIES_Normalized/'):
    """This function enables to read the content of the folder processed of root folder of dataset
    and write the .txt file that hosts the list of the names of the graphs that are needed
    to already ecist within processed folder in order to skip process method during dataset instantiation"""
    items = os.listdir(folder_path)
    items = [item for item in items if item not in ['pre_filter.pt', 'pre_transform.pt']]
    with open(os.path.join(root, 'ref_data_list.txt'), 'w') as file:
        # Write each string in the list to the file
        for item in items:
            file.write("%s\n" % item)
    return

def get_subgraph_from_nbunch(g: SimpleCenterlineGraph, node_list: list) -> SimpleCenterlineGraph:
    
    """Custum function to generate a graph of class SimpleCenterlineGraph that is the subgraph deriving from a given list of nodes"""
    SG=g.__class__(**g.graph) #setting the graph-level attributes copied from original graph 
    SG.graph["image_id"] += " - subgraph" #changeing the name at graph level
    SG.add_nodes_from((n, g.nodes[n]) for n in node_list)
    SG.add_edges_from((n, nbr, d)
                    for n, nbrs in g.adj.items() if n in node_list
                    for nbr, d in nbrs.items() if nbr in node_list)
    SG.graph.update(g.graph)
    return SG

def from_networkx(
    G: Any,
    group_node_attrs: Optional[Union[List[str], all]] = None,
    group_edge_attrs: Optional[Union[List[str], all]] = None,
) -> Data:
    r"""Converts a :obj:`networkx.Graph` or :obj:`networkx.DiGraph` to a
    :class:`torch_geometric.data.Data` instance.

    Args:
        G (networkx.Graph or networkx.DiGraph): A networkx graph.
        group_node_attrs (List[str] or all, optional): The node attributes to
            be concatenated and added to :obj:`data.x`. (default: :obj:`None`)
        group_edge_attrs (List[str] or all, optional): The edge attributes to
            be concatenated and added to :obj:`data.edge_attr`.
            (default: :obj:`None`)

    .. note::

        All :attr:`group_node_attrs` and :attr:`group_edge_attrs` values must
        be numeric. 
        for 'topology' attribute, a one hot encoding is performed' since the values are not ordinal

    Examples:
        >>> edge_index = torch.tensor([
        ...     [0, 1, 1, 2, 2, 3],
        ...     [1, 0, 2, 1, 3, 2],
        ... ])
        >>> data = Data(edge_index=edge_index, num_nodes=4)
        >>> g = to_networkx(data)
        >>> # A `Data` object is returned
        >>> from_networkx(g)
        Data(edge_index=[2, 6], num_nodes=4)
    """

    G = G.to_directed() if not nx.is_directed(G) else G

    mapping = dict(zip(G.nodes(), range(G.number_of_nodes())))
    edge_index = torch.empty((2, G.number_of_edges()), dtype=torch.long)
    for i, (src, dst) in enumerate(G.edges()):
        edge_index[0, i] = mapping[src]
        edge_index[1, i] = mapping[dst]

    data = defaultdict(list)

    if G.number_of_nodes() > 0:
        node_attrs = list(next(iter(G.nodes(data=True)))[-1].keys())
    else:
        node_attrs = {}

    if G.number_of_edges() > 0:
        edge_attrs = list(next(iter(G.edges(data=True)))[-1].keys())
    else:
        edge_attrs = {}

    for i, (_, feat_dict) in enumerate(G.nodes(data=True)):
        if set(feat_dict.keys()) != set(node_attrs):
            raise ValueError('Not all nodes contain the same attributes')
        for key, value in feat_dict.items():
            if key == 'topology': #one hot encoding of the topology attribute
                value = one_hot_encoding_topologies(value)
            data[str(key)].append(value)

    for i, (_, _, feat_dict) in enumerate(G.edges(data=True)):
        if set(feat_dict.keys()) != set(edge_attrs):
            raise ValueError('Not all edges contain the same attributes')
        for key, value in feat_dict.items():
            key = f'edge_{key}' if key in node_attrs else key
            data[str(key)].append(value)

    for key, value in G.graph.items():
        if key == 'node_default' or key == 'edge_default':
            continue  # Do not load default attributes.
        key = f'graph_{key}' if key in node_attrs else key
        data[str(key)] = value

    for key, value in data.items():
        if isinstance(value, (tuple, list)) and isinstance(value[0], Tensor):
            data[key] = torch.stack(value, dim=0)
        else:
            try:
                data[key] = torch.tensor(value)
            except (ValueError, TypeError, RuntimeError):
                pass

    data['edge_index'] = edge_index.view(2, -1)
    data = Data.from_dict(data)

    if group_node_attrs is all:
        group_node_attrs = list(node_attrs)
    if group_node_attrs is not None:
        xs = []
        for key in group_node_attrs:
            x = data[key]
            x = x.view(-1, 1) if x.dim() <= 1 else x
            xs.append(x)
            del data[key]
        data.x = torch.cat(xs, dim=-1)

    if group_edge_attrs is all:
        group_edge_attrs = list(edge_attrs)

    if group_edge_attrs is not None:
        xs = []
        for key in group_edge_attrs:
            key = f'edge_{key}' if key in node_attrs else key
            x = data[key]
            x = x.view(-1, 1) if x.dim() <= 1 else x
            xs.append(x)
            del data[key]
        data.edge_attr = torch.cat(xs, dim=-1)

    if data.x is None and data.pos is None:
        data.num_nodes = G.number_of_nodes()

    return data


def one_hot_encoding_topologies(value: str) -> torch.Tensor:
    assert any(value.upper() == topology.name for topology in ArteryNodeTopology), f"Expected value to be part of ArteryNodeTopology, got {value}"
    topology_number = len(ArteryNodeTopology)
    topology_mask=[value.upper() == topology.name for topology in ArteryNodeTopology]
    topology_one_hot = torch.zeros(topology_number)
    topology_one_hot[topology_mask] = 1
    return topology_one_hot

def visualize_graphs_from_folder(folder='/home/erikfer/GNN_project/DATA/SPLITTED_ARTERIES_DATA/', from_id=0) -> None:
    """This function enables to visualize the graphs in networkx format contained in the folder
    folder must be the path to the directory 'raw' of structure:
    DATA_PATH
    folder
        raw
            left_artery
                ASOCA
                    asoca_graph_1_left.gml
                    asoca_graph_2_left.gml
                    ...
                CAT08
                    cat08_graph_1_left.gml
                    cat08_graph_2_left.gml
                    ...
            right_artery
                ASOCA
                    asoca_graph_1_right.gml
                    asoca_graph_2_right.gml
                    ...
                CAT08
                    cat08_graph_1_right.gml
                    cat08_graph_2_right.gml
                    ...
            graph_annotations.json"""
    
    with open(os.path.join(folder, 'raw/graphs_annotation.json'), "r") as json_file:
            data = json.load(json_file)

    cat_id_to_name = {category["id"]: category["name"] for category in data["categories"]}
    for i, graph_raw in enumerate(data["graphs"][from_id:]):
        file_name=graph_raw["file_name"]
        category_id= cat_id_to_name[graph_raw["category_id"]]
        g = hcatnetwork.io.load_graph(file_path=os.path.join(folder, file_name),
                                      output_type=hcatnetwork.graph.SimpleCenterlineGraph)
        hcatnetwork.draw.draw_simple_centerlines_graph_2d(g, backend="networkx")

def make_ostia_origin_and_normalize(side_graph, normalize=True):
    """ This function enables to shift the origin of the graph to the ostium node and to normalize the coordinates of the nodes
    obs: ostium_id is not the node index but the id of the ostium in the graph annotation file"""


    # data = defaultdict(list)

    # if side_graph.number_of_nodes() > 0:
    #     node_attrs = list(next(iter(side_graph.nodes(data=True)))[-1].keys())
    # else:
    #     node_attrs = {}
    
    coord_list = list()
    for i, (name, feat_dict) in enumerate(side_graph.nodes(data=True)):
        coord_list.append(np.array([feat_dict['x'], feat_dict['y'], feat_dict['z']], dtype=np.float32))
        if feat_dict['topology'].name == 'OSTIUM':
            xyz_ostium = np.array([side_graph.nodes[name]['x'], side_graph.nodes[name]['y'], \
                                    side_graph.nodes[name]['z']], dtype=np.float32)

    assert xyz_ostium is not None, "The ostium node is not found"
    assert len(coord_list) == side_graph.number_of_nodes(), "The number of coordinates is not equal to the number of nodes"
    coord_matrix = np.array(coord_list, dtype=np.float32)
    coord_matrix_shifted = coord_matrix - xyz_ostium
    
    if normalize:
        min_coords = np.min(coord_matrix_shifted, axis=0)
        max_coords = np.max(coord_matrix_shifted, axis=0)
        coords = (coord_matrix_shifted - min_coords) / (max_coords - min_coords)
        coord_matrix_shifted = coords
    
    for i, (name, feat_dict) in enumerate(side_graph.nodes(data=True)):
        side_graph.nodes[name]['x'] = float(coord_matrix_shifted[i, 0])
        side_graph.nodes[name]['y'] = float(coord_matrix_shifted[i, 1])
        side_graph.nodes[name]['z'] = float(coord_matrix_shifted[i, 2])

    #update edge length: weight and euclidean distance
    for i, (u, v, feat_dict) in enumerate(side_graph.edges(data=True)):
        u_xyz_coord=np.array([side_graph.nodes[u]['x'], side_graph.nodes[u]['y'], side_graph.nodes[u]['z']], dtype=np.float32)
        v_xyz_coord=np.array([side_graph.nodes[v]['x'], side_graph.nodes[v]['y'], side_graph.nodes[v]['z']], dtype=np.float32)
        #update the weight of the edge between u and v
        # print(graph[u][v]['euclidean_distance'])
        side_graph[u][v]['weight'] = float(np.linalg.norm(u_xyz_coord-v_xyz_coord))
        side_graph[u][v]['euclidean_distance'] = float(np.linalg.norm(u_xyz_coord-v_xyz_coord))

    return side_graph

def test_make_ostia_origin_and_normalize():
    # Create a test graph
    side_graph = nx.Graph()
    side_graph.add_node('A', x='0.5', y='0.5', z='0.5', topology=ArteryNodeTopology.SEGMENT)
    side_graph.add_node('B', x='0.2', y='0.8', z='0.3', topology=ArteryNodeTopology.OSTIUM)
    side_graph.add_node('C', x='0.7', y='0.3', z='0.9', topology=ArteryNodeTopology.ENDPOINT)

    # Call the function
    result = make_ostia_origin_and_normalize(side_graph)

    # Check if all values of x, y, z are between 0 and 1
    for name, feat_dict in result.nodes(data=True):
        x = float(feat_dict['x'])
        y = float(feat_dict['y'])
        z = float(feat_dict['z'])
        assert 0 <= x <= 1, f"Value of x ({x}) is not between 0 and 1"
        assert 0 <= y <= 1, f"Value of y ({y}) is not between 0 and 1"
        assert 0 <= z <= 1, f"Value of z ({z}) is not between 0 and 1"

    # Check if OSTIUM is at coordinates 0, 0, 0
    ostium_node = result.nodes['B']
    assert float(ostium_node['x']) == 0 or float(ostium_node['y']) == 0 or float(ostium_node['z']) == 0, \
    'OSTIUM node is not in a starting coordinate i.e. it is 0 in at least one coordinate'
    print('Test passed')
