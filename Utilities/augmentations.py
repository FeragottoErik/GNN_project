import networkx as nx
import numpy as np
import os
import json
import hcatnetwork
import random
import warnings
from Utilities.custom_functions import get_subgraph_from_nbunch

def apply_augmentation_to_graph(graph: nx.Graph, augmentation: list[str], prob=0.5, **kwargs):
    """Applies the specified augmentation to the given graph and returns the augmented graph.
    Available augmentations:
    - 'random_node_deletion': deletes a random node from the graph
    - 'random_edge_deletion': deletes a random edge from the graph
    - 'random_node_addition': adds a random node to the graph
    - 'random_edge_addition': adds a random edge to the graph
    - 'random_node_attribute_change': changes the attribute of a random node in the graph
    - 'random_edge_attribute_change': changes the attribute of a random edge in the graph
    - 'random_node_attribute_deletion': deletes an attribute from a random node in the graph
    - 'random_edge_attribute_deletion': deletes an attribute from a random edge in the graph
    - 'random_noise_on_node_position': adds random noise to the position of a random node in the graph
    - 'chunk_random_segment': chunks a random segment of the graph
    - 'random_graph_portion_selection': selects a random portion of the graph pointing out a random node and 
                                        all the nodes within an euclidean distance from it

    Args:
        graph (nx.Graph): the graph to augment
        augmentation (list[str]): list of augmentations to apply to the graph
        prob (float, optional): probability of applying the augmentation. Defaults to 0.5.
    Returns:
        nx.Graph: the augmented graph
    """

    # Check if the augmentation should be applied
    if np.random.random() > prob:
        return graph

    # Apply the augmentation
    for aug in augmentation:
        if aug == 'random_node_deletion':
            graph = random_node_deletion(graph)
        elif aug == 'random_edge_deletion':
            graph = random_edge_deletion(graph)
        elif aug == 'random_node_addition':
            graph = random_node_addition(graph)
        elif aug == 'random_edge_addition':
            graph = random_edge_addition(graph)
        elif aug == 'random_node_attribute_change':
            graph = random_node_attribute_change(graph)
        elif aug == 'random_edge_attribute_change':
            graph = random_edge_attribute_change(graph)
        elif aug == 'random_node_attribute_deletion':
            graph = random_node_attribute_deletion(graph)
        elif aug == 'random_edge_attribute_deletion':
            graph = random_edge_attribute_deletion(graph)
        elif aug == 'random_noise_on_node_position':
            graph = random_noise_on_node_position(graph)
        elif aug == 'chunk_random_segment':
            graph = chunk_random_segment(graph)
        elif aug == 'random_graph_portion_selection':
            graph = random_graph_portion_selection(graph)
        else:
            raise ValueError("Augmentation not recognized")

    return graph

def random_node_deletion(graph: nx.Graph):
    """Deletes a random node from the graph

    Args:
        graph (nx.Graph): the graph to augment

    Returns:
        nx.Graph: the augmented graph
    """
    # Select a random node
    node = np.random.choice(list(graph.nodes))

    # Delete the node
    graph.remove_node(node)

    return graph

def random_edge_deletion(graph: nx.Graph):
    """Deletes a random edge from the graph

    Args:
        graph (nx.Graph): the graph to augment

    Returns:
        nx.Graph: the augmented graph
    """
    # Select a random edge
    edge = np.random.choice(list(graph.edges))

    # Delete the edge
    graph.remove_edge(*edge)

    return graph

def random_node_addition(graph: nx.Graph):
    """Adds a random node to the graph

    Args:
        graph (nx.Graph): the graph to augment

    Returns:
        nx.Graph: the augmented graph
    """
    # Add a new node
    graph.add_node(graph.number_of_nodes())

    return graph

def random_edge_addition(graph: nx.Graph):
    """Adds a random edge to the graph

    Args:
        graph (nx.Graph): the graph to augment

    Returns:
        nx.Graph: the augmented graph
    """
    # Select a random node
    node = np.random.choice(list(graph.nodes))

    # Add a new edge
    graph.add_edge(node, graph.number_of_nodes())

    return graph

def random_node_attribute_change(graph: nx.Graph):
    """Changes the attribute of a random node in the graph

    Args:
        graph (nx.Graph): the graph to augment

    Returns:
        nx.Graph: the augmented graph
    """
    # Select a random node
    node = np.random.choice(list(graph.nodes))

    # Change the attribute of the node
    graph.nodes[node]['x'] = np.random.random()
    graph.nodes[node]['y'] = np.random.random()
    graph.nodes[node]['z'] = np.random.random()

    return graph

def random_noise_on_node_position(graph: nx.Graph, max_shift=None):
    """Adds random noise to the position of a random node in the graph updating as needed the edge weights
    and euclidean distances. The maximal random shift in position of the node is half of the shortest edge length.
    Otherwise it can couse overlapping of edges or loss of spatial sequentiality.

    Args:
        graph (nx.Graph): the graph to augment
        max_shift (float, optional): maximal random shift in position of the node. Defaults to None.

    Returns:
        nx.Graph: the augmented graph
    """
    max_dist=0
    min_dist=1000
    for edge in list(graph.edges):
        weight = (graph[edge[0]][edge[1]]['weight'])
        if weight > max_dist:
            max_dist=weight
        if weight < min_dist:
            min_dist=weight

    #maximal random shift in position must be half of the shortest edge length
    #otherwise it can couse overlapping of edges or loss of spatial sequentiality
    if max_shift is None or max_shift > min_dist/2:
        if max_shift is not None and (max_shift > min_dist/2):
            warnings.warn("The maximal random shift in position of the node is greater than the half of the shortest edge length. The maximal random shift in position of the node is set to half of the shortest edge length")
        max_shift = min_dist/2

    weight = max_shift

    # Add random noise to the position of the node 
    for i, (name, feat_dict) in enumerate(graph.nodes(data=True)):
        xyz_coord=np.array([feat_dict['x'], feat_dict['y'], feat_dict['z']], dtype=np.float32)
        augment_array = np.random.uniform(low=-weight, high=weight, size=3)
        augmented_coord = xyz_coord + augment_array
        graph.nodes[name]['x'] = augmented_coord[0]
        graph.nodes[name]['y'] = augmented_coord[1]
        graph.nodes[name]['z'] = augmented_coord[2]


    max_lenght=0
    min_lenght=1000
    # scroll through edge list and update the weight as the abs difference in euclidean distance between the connected nodes
    for i, (u, v, feat_dict) in enumerate(graph.edges(data=True)):
        u_xyz_coord=np.array([graph.nodes[u]['x'], graph.nodes[u]['y'], graph.nodes[u]['z']], dtype=np.float32)
        v_xyz_coord=np.array([graph.nodes[v]['x'], graph.nodes[v]['y'], graph.nodes[v]['z']], dtype=np.float32)
        #update the weight of the edge between u and v
        # print(graph[u][v]['euclidean_distance'])
        graph[u][v]['weight'] = np.linalg.norm(u_xyz_coord-v_xyz_coord)
        graph[u][v]['euclidean_distance'] = np.linalg.norm(u_xyz_coord-v_xyz_coord)
        # print(graph[u][v]['euclidean_distance'])
        # print('\n\n')
        if graph[u][v]['weight'] < min_lenght:
            min_lenght=graph[u][v]['weight']
        if graph[u][v]['weight'] > max_lenght:
            max_lenght=graph[u][v]['weight']

    assert max_lenght < max_dist+min_dist, "The maximal edge length is greater than the sum of the greater edge length and the 2 times the maximal variation in position of the nodes"
    assert min_lenght > 0, "The minimal edge length is negative or zero"

    return graph


def random_graph_portion_selection(g: nx.Graph, neigh_dist: int | str = 'random', start_node: str = 'random') -> nx.Graph:
    """Given a graph, selects with different strategies a portion of it and returns the subgraph.
    The subgraph is selected in such a way that the distance from the target node to the farthest node in the subgraph is equal to neigh_dist.
    If neigh_dist is greater than the longest segment in one direction, all nodes in that direction are selected.
    If start_node is 'random' a random node is selected as target_node.
    If neigh_dist is 'random' a random integer between 50 and the longest segment in one direction given the target_node is selected.
    In the whole dataset, the graph with the longest segment between an OSTIUM and an ENDPOINT counts a path 571 nodes long, while graph where the longest path from OSTIUM 
    to the most far endpoint is the shortest among all graphs, is 167 nodes long. 

    Args:
        g (nx.Graph): the graph to augment
        neigh_dist (int | str, optional): the distance from the target node to select the subgraph. Defaults to 'random'.
        start_node (str, optional): the node from which the distance is computed. Defaults to 'random'. Can be one of: 'OSTIUM', 'ENDPOINT', 'INTERSECTION', 'random'
    
    Returns:
        nx.Graph: the subgraph"""

    MIN_NEIGH_DIST = 50
    #MAX_NEIGH_DIST = len(g.nodes)//2

    valid_start_nodes = ['OSTIUM', 'ENDPOINT', 'INTERSECTION', 'random']
    assert start_node in valid_start_nodes, f"The start_node parameter must be one of {valid_start_nodes}"
    

    if start_node == 'random':
        target_node = random.choice(list(g.nodes))

    else:
        for i, (name, feat_dict) in enumerate(g.nodes(data=True)):
            if feat_dict['topology'].name == start_node: #if the start_node is ENDPONT or INTERSECTION first occurence is selected
                target_node = name
                break
    
    #compute the longhest segment from target_node in any direction in terms of number of nodes
    max_segment_length = max([len(path) for path in nx.single_source_shortest_path(g, target_node).values()])
    MAX_NEIGH_DIST=max_segment_length

    if isinstance(neigh_dist, int) and not (MIN_NEIGH_DIST <= neigh_dist <= MAX_NEIGH_DIST):
        warnings.warn(f"The neigh_dist parameter should be an integer between {MIN_NEIGH_DIST} and {MAX_NEIGH_DIST}")
    
    elif isinstance(neigh_dist, str):
        assert neigh_dist == 'random', "The neigh_dist parameter must be either an integer or the string 'random'"
        neigh_dist = random.randint(MIN_NEIGH_DIST, MAX_NEIGH_DIST)

    #create a list that contains all the nodes of the graph that are at a distance <= neigh_dist from the target_node 
    #if neigh_dist is greater than the longest segment in one direction, all nodes in that direction are selected

    nodes_to_keep = [target_node]
    for i in range(neigh_dist):
        nodes_to_keep.extend(list(nx.single_source_shortest_path_length(g, target_node, cutoff=i).keys()))
    
    #create a subgraph containing only the nodes in nodes_to_keep
    subgraph = get_subgraph_from_nbunch(g, nodes_to_keep)

    return subgraph



if __name__ == "__main__":
    # Set up the dataset
    #dataset = ArteryGraphDataset(root='/home/erikfer/GNN_project/DATA/SPLITTED_ARTERIES_DATA/', ann_file='graphs_annotation.json')
    # Split the dataset into training and test sets with 80-20 splitting
    folder='/home/erikfer/GNN_project/DATA/SPLITTED_ARTERIES_Normalized/'
    with open(os.path.join(folder, 'raw/graphs_annotation.json'), "r") as json_file:
        data = json.load(json_file)

    cat_id_to_name = {category["id"]: category["name"] for category in data["categories"]}
    max_dist_from_ostium=list()
    for i, graph_raw in enumerate(data["graphs"][0:]):
        file_name=graph_raw["file_name"]
        category_id= cat_id_to_name[graph_raw["category_id"]]
        g = hcatnetwork.io.load_graph(file_path=os.path.join(folder, file_name),
                                      output_type=hcatnetwork.graph.SimpleCenterlineGraph)
        #hcatnetwork.draw.draw_simple_centerlines_graph_2d(g, backend="networkx")
        
        #aug_graph = random_noise_on_node_position(g)
        #hcatnetwork.draw.draw_simple_centerlines_graph_2d(aug_graph, backend="networkx")

        aug_graph= random_graph_portion_selection(g, 'random', 'OSTIUM')
        hcatnetwork.draw.draw_simple_centerlines_graph_2d(aug_graph, backend="networkx")
    #print(max(max_dist_from_ostium), min(max_dist_from_ostium))