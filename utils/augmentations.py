import networkx as nx
import numpy as np
import os
import json
import hcatnetwork
import random
import warnings
from utils.custom_functions import get_subgraph_from_nbunch, find_connected_nodes, generate_fake_vessel, generate_points_on_sphere
from hcatnetwork.node import ArteryNodeTopology
from HearticDatasetManager.affine import get_affine_3d_rotation_around_vector
from HearticDatasetManager.affine import apply_affine_3d
from typing import Any, Dict, List, Union

def apply_augmentation_to_graph(graph: nx.Graph, augmentation: list[str], prob=0.5, n_changes=1,**kwargs):
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
    - 'random_graph_affine_transformation': applies a random affine transformation to the graph
    - 'all': applies all the augmentations listed above

    Args:
        graph (nx.Graph): the graph to augment
        augmentation (list[str]): list of augmentations to apply to the graph
        prob (float, optional): probability of applying each one of the augmentations. Defaults to 0.5.
    Returns:
        nx.Graph: the augmented graph
    """

    """Use this if you want to apply the augmentation with a certain probability or return the unmodified graph, 
    otherwise the augmentation is always applied with a certain probability augmentation-wise"""
    # Check if the augmentation should be applied
    # if np.random.random() > prob:
    #     return graph

    # Apply all selected augmentations
    if 'all' in augmentation:
        augmentation = ['random_graph_portion_selection', 'random_noise_on_node_position', 'random_graph_affine_transformation']
        
    #the prob to not apply any of the augmentations is (1-prob)**len(augmentation)
    for aug in augmentation:
        if np.random.random() > prob: #the prob is not to apply or not any augmentation to the graph, but to apply or not the specific augmentation
            continue
        # if aug == 'random_node_deletion':
        #     graph = random_node_deletion(graph, n_changes)
        # elif aug == 'random_edge_deletion':
        #     graph = random_edge_deletion(graph, n_changes)
        # elif aug == 'random_node_addition':
        #     graph = random_node_addition(graph, n_changes)
        # elif aug == 'random_edge_addition':
        #     graph = random_edge_addition(graph, n_changes)
        # elif aug == 'random_node_attribute_change':
        #     graph = random_node_attribute_change(graph, n_changes)
        if aug == 'random_noise_on_node_position':
            graph = random_noise_on_node_position(graph)
        elif aug == 'random_graph_portion_selection':
            graph = random_graph_portion_selection(graph, 'random', 'OSTIUM')
        elif aug == 'random_graph_affine_transformation':
            graph = random_graph_affine_transformation(graph, 'OSTIUM')
        else:
            raise ValueError("Augmentation not recognized")

    return graph

def random_node_deletion(graph: nx.Graph, n_nodes: int = 1):
    """Deletes a random node from the graph

    Args:
        graph (nx.Graph): the graph to augment

    Returns:
        nx.Graph: the augmented graph
    """
    for _ in range(n_nodes):
        # Select a random node
        node = random.choice(list(graph.nodes))

        #remove the edges that have as one of the nodes the node that has been deleted
        connected_edges = list(graph.edges(node))
        graph.remove_edges_from(connected_edges)

        # Delete the node
        graph.remove_node(node)

    return graph

def random_edge_deletion(graph: nx.Graph, n_edges: int = 1):
    """Deletes a random edge from the graph

    Args:
        graph (nx.Graph): the graph to augment

    Returns:
        nx.Graph: the augmented graph
    """
    for _ in range(n_edges):
        # Select a random edge
        edge = random.choice(list(graph.edges))

        # Delete the edge
        graph.remove_edge(*edge)

    return graph

def random_node_addition(graph: nx.Graph, n_nodes: int = 1):
    """Adds a random node to the graph

    Args:
        graph (nx.Graph): the graph to augment

    Returns:
        nx.Graph: the augmented graph
    """
    for _ in range(n_nodes):
        #compute maximal and minimal values for x,y,z coordinates in all nodes of the graph
        max_x=0
        min_x=1000
        max_y=0
        min_y=1000
        max_z=0
        min_z=1000
        for i, (name, feat_dict) in enumerate(graph.nodes(data=True)):
            if feat_dict['x'] > max_x:
                max_x=feat_dict['x']
            if feat_dict['x'] < min_x:
                min_x=feat_dict['x']
            if feat_dict['y'] > max_y:
                max_y=feat_dict['y']
            if feat_dict['y'] < min_y:
                min_y=feat_dict['y']
            if feat_dict['z'] > max_z:
                max_z=feat_dict['z']
            if feat_dict['z'] < min_z:
                min_z=feat_dict['z']
        
        # Select a random set of coordinates between min and max of the target coordinate
        x=np.random.uniform(low=min_x, high=max_x)
        y=np.random.uniform(low=min_y, high=max_y)
        z=np.random.uniform(low=min_z, high=max_z)

        #find the first and the second closest nodes to the random coordinates
        closest_node_1 = min(graph.nodes, key=lambda n: np.linalg.norm(np.array\
                        ([graph.nodes[n]['x'], graph.nodes[n]['y'], graph.nodes[n]['z']], dtype=np.float32)-np.array([x, y, z], dtype=np.float32)))
        dist_node_1 = float(np.linalg.norm(np.array\
                        ([graph.nodes[closest_node_1]['x'], graph.nodes[closest_node_1]['y'], graph.nodes[closest_node_1]['z']], dtype=np.float32)-np.array([x, y, z], dtype=np.float32)))
        closest_node_2 = min(graph.nodes, key=lambda n: np.linalg.norm(np.array\
                        ([graph.nodes[n]['x'], graph.nodes[n]['y'], graph.nodes[n]['z']], dtype=np.float32)-np.array([x, y, z], dtype=np.float32)) if n != closest_node_1 else np.inf)
        dist_node_2 = float(np.linalg.norm(np.array\
                        ([graph.nodes[closest_node_2]['x'], graph.nodes[closest_node_2]['y'], graph.nodes[closest_node_2]['z']], dtype=np.float32)-np.array([x, y, z], dtype=np.float32)))
        
        #r is the mean between the 'r' value of closest_node_1 and the 'r' value of closest_node_2
        r = float((graph.nodes[closest_node_1]['r']+graph.nodes[closest_node_2]['r'])/2)
        #topology is SEGMENT (imposed)
        topology = ArteryNodeTopology.SEGMENT
        #t is 0 (imposed)
        t = float(0)
        #side is the 'side' attribute of closest_node_1
        side = graph.nodes[closest_node_1]['side']


        #add two edges and a node: the node has coordinates x,y,z, the edges connect x,y,z node to node_1 and node_2 with weight and euclidean_distance equal to the distance between the nodes
        graph.add_node(str(len(graph.nodes)), x=x, y=y, z=z, r=r, topology=topology, t=t, side=side)
        graph.add_edge(str(len(graph.nodes)-1), closest_node_1, weight=dist_node_1, euclidean_distance=dist_node_1)
        graph.add_edge(str(len(graph.nodes)-1), closest_node_2, weight=dist_node_2, euclidean_distance=dist_node_2)
        print(graph.nodes[str(len(graph.nodes)-1)])

        # Add a new node to the graph and assign to 'x', 'y' and 'z' a random value between min and max of the target coordinate
        #graph.add_node(len(graph.nodes), x=np.random.uniform(low=min_x, high=max_x), y=np.random.uniform(low=min_y, high=max_y), z=np.random.uniform(low=min_z, high=max_z))

    return graph

def random_edge_addition(graph: nx.Graph, n_edges: int = 1):
    """Adds a random edge to the graph

    Args:
        graph (nx.Graph): the graph to augment

    Returns:
        nx.Graph: the augmented graph
    """
    for _ in range(n_edges):
        # Select a random node
        u = random.choice(list(graph.nodes))
        v = random.choice(list(graph.nodes)) #.pop(list(graph.nodes).index(u)))

        # Add a new edge between the two nodes selected and assign to 'weight' and 'euclidian_distance' the euclidean distance between the two nodes
        u_xyz_coord=np.array([graph.nodes[u]['x'], graph.nodes[u]['y'], graph.nodes[u]['z']], dtype=np.float32)
        v_xyz_coord=np.array([graph.nodes[v]['x'], graph.nodes[v]['y'], graph.nodes[v]['z']], dtype=np.float32)
        dist = float(np.linalg.norm(u_xyz_coord-v_xyz_coord))
        graph.add_edge(u, v, weight=dist, euclidean_distance=dist)


    return graph

def random_node_attribute_change(graph: nx.Graph, n_nodes: int = 1):
    """Changes the attribute of a random node in the graph

    Args:
        graph (nx.Graph): the graph to augment

    Returns:
        nx.Graph: the augmented graph
    """

    #compute maximal and minimal values for x,y,z coordinates in all nodes of the graph
    max_x=0
    min_x=1000
    max_y=0
    min_y=1000
    max_z=0
    min_z=1000
    for i, (name, feat_dict) in enumerate(graph.nodes(data=True)):
        if feat_dict['x'] > max_x:
            max_x=feat_dict['x']
        if feat_dict['x'] < min_x:
            min_x=feat_dict['x']
        if feat_dict['y'] > max_y:
            max_y=feat_dict['y']
        if feat_dict['y'] < min_y:
            min_y=feat_dict['y']
        if feat_dict['z'] > max_z:
            max_z=feat_dict['z']
        if feat_dict['z'] < min_z:
            min_z=feat_dict['z']

    for _ in range(n_nodes):
        # Select a random node
        node = np.random.choice(list(graph.nodes))
        # Change the attribute of the node so as to assign e random value between min and max of the target coordinate
        graph.nodes[node]['x'] = np.random.uniform(low=min_x, high=max_x)
        graph.nodes[node]['y'] = np.random.uniform(low=min_y, high=max_y)
        graph.nodes[node]['z'] = np.random.uniform(low=min_z, high=max_z)

        #list the edges that have as one of the nodes the node that has been modified
        connected_edges = list(graph.edges(node))

        #update the euclidean distance and the weight of the edges between the node and the connected nodes
        for u, v in connected_edges:
            u_xyz_coord=np.array([graph.nodes[u]['x'], graph.nodes[u]['y'], graph.nodes[u]['z']], dtype=np.float32)
            v_xyz_coord=np.array([graph.nodes[v]['x'], graph.nodes[v]['y'], graph.nodes[v]['z']], dtype=np.float32)
            graph[u][v]['euclidean_distance'] = float(np.linalg.norm(u_xyz_coord-v_xyz_coord))
            graph[u][v]['weight'] = float(np.linalg.norm(u_xyz_coord-v_xyz_coord))

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
        graph.nodes[name]['x'] = float(augmented_coord[0])
        graph.nodes[name]['y'] = float(augmented_coord[1])
        graph.nodes[name]['z'] = float(augmented_coord[2])


    max_lenght=0
    min_lenght=1000
    # scroll through edge list and update the weight as the abs difference in euclidean distance between the connected nodes
    for i, (u, v, feat_dict) in enumerate(graph.edges(data=True)):
        u_xyz_coord=np.array([graph.nodes[u]['x'], graph.nodes[u]['y'], graph.nodes[u]['z']], dtype=np.float32)
        v_xyz_coord=np.array([graph.nodes[v]['x'], graph.nodes[v]['y'], graph.nodes[v]['z']], dtype=np.float32)
        #update the weight of the edge between u and v
        # print(graph[u][v]['euclidean_distance'])
        graph[u][v]['weight'] = float(np.linalg.norm(u_xyz_coord-v_xyz_coord))
        graph[u][v]['euclidean_distance'] = float(np.linalg.norm(u_xyz_coord-v_xyz_coord))
        # print(graph[u][v]['euclidean_distance'])
        # print('\n\n')
        if graph[u][v]['weight'] < min_lenght:
            min_lenght=graph[u][v]['weight']
        if graph[u][v]['weight'] > max_lenght:
            max_lenght=graph[u][v]['weight']

    # assert max_lenght <= max_dist+min_dist, f"The maximal edge length {max_lenght}  is greater than the sum of the greater edge length {max_dist}  and the 2 times the maximal variation in position of the nodes {min_dist} "
    # assert min_lenght > 0, "The minimal edge length is negative or zero"

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
            try:
                if feat_dict['topology'].name == start_node: #if the start_node is ENDPONT or INTERSECTION first occurence is selected
                    target_node = name
                    break
            except: #in case topology is not a ArteryNodeTopology enum but a string
                if feat_dict['topology'] == start_node:
                    target_node = name
                    break

    if target_node is None:
        target_node = random.choice(list(g.nodes)) #accounts for the case in which the ostium or a node of interest has been 
        #delated by other augmentations
    
    #compute the longhest segment from target_node in any direction in terms of number of nodes
    max_segment_length = max([len(path) for path in nx.single_source_shortest_path(g, target_node).values()])
    MAX_NEIGH_DIST=max_segment_length
    if MAX_NEIGH_DIST<MIN_NEIGH_DIST:
        MIN_NEIGH_DIST=MAX_NEIGH_DIST

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

def random_graph_affine_transformation(g: nx.Graph, app_point: str | int = 'OSTIUM', vect: Union[np.array, list]= None, alpha_rad: float = None) -> nx.Graph:
    if isinstance(app_point, str):
        assert app_point in ['OSTIUM', 'random'], "The app_point parameter must be either an integer or the string 'OSTIUM' or the string 'random'"
    else:
        assert isinstance(app_point, int), "The app_point parameter must be either an integer or the string 'OSTIUM' or the string 'random'"

    if vect is not None and len(vect)!=3:
        raise ValueError("The vect parameter must be a list or a numpy array of length 3 as the 3 dimensions x,y,z")

    app_point_name = None
    if app_point == 'OSTIUM':
        #obs: if the graph is normalized with Utils.custom_functions.make_ostia_origin_and_normalize, the ostium is always in (0,0,0)
        for i, (name, feat_dict) in enumerate(g.nodes(data=True)):
            try:
                if feat_dict['topology'].name == app_point: 
                    app_point_name = name
                    break
            except:
                if feat_dict['topology'] == app_point: #accounts for the case in which the typology is not an ArteryNodeTopology enum but a string
                    app_point_name = name
                    break

        if app_point_name is None:
            warnings.warn("The app_point parameter has no been found, it may indicate that there is no OSTIUM \
                          in the graph, application point of the vector is selected randomly")
    elif isinstance(app_point, int):
        if app_point not in list(g.nodes):
            warnings.warn("The app_point parameter must be an integer within the graph nodes names")
        else:
            app_point_name = str(app_point) #node names are strings not integers
            
    if app_point == 'random' or app_point_name is None:
        #the point of application of the vector for the affine transformation is a random node of the graph
        app_point_name = random.choice(list(g.nodes))

    assert app_point_name is not None, "The app_point parameter has no been found error in the implementation code"

    xyz_app_point = np.array([g.nodes[app_point_name]['x'], g.nodes[app_point_name]['y'], g.nodes[app_point_name]['z']], dtype=np.float32)

    if vect is not None:
        vect = np.array(vect, dtype=np.float32)
    else:
        vect = np.random.uniform(low=-1, high=1, size=3) #random vector of length 1 (versor) with random direction
    
    #create a matrix with all graph points x,y,z as rows
    xyz_points = np.zeros((len(g.nodes), 3), dtype=np.float32)
    for i, (name, feat_dict) in enumerate(g.nodes(data=True)):
        xyz_points[i, 0] = float(feat_dict['x'])
        xyz_points[i, 1] = float(feat_dict['y'])
        xyz_points[i, 2] = float(feat_dict['z'])
    #apply the affine transformation to the graph
    if alpha_rad is None:
        alpha_rad = np.random.uniform(low=-np.pi, high=np.pi)
    else:
        assert -np.pi <= float(alpha_rad) <= np.pi, "The alpha_rad parameter must be a float between -pi and pi since it is in radians"

    transform = get_affine_3d_rotation_around_vector(xyz_app_point, vect, alpha_rad)
    transformed_xyz_points = apply_affine_3d(transform, xyz_points.T).T
    assert transformed_xyz_points.shape == xyz_points.shape, "The shape of the transformed points is different from the original points"

    #reassign the new x,y,z coordinates to the graph nodes
    for i, (name, feat_dict) in enumerate(g.nodes(data=True)):
        g.nodes[name]['x'] = float(transformed_xyz_points[i, 0])
        g.nodes[name]['y'] = float(transformed_xyz_points[i, 1])
        g.nodes[name]['z'] = float(transformed_xyz_points[i, 2])

    #the euclidian distance between nodes is preserved because the transformation is affine -> no need to update edge attrs
    return g

def flip_graph_upside_down(g: nx.Graph) -> nx.Graph:
    """Flips the graph upside down with respect to the x axis (i.e. y and z coordinates are multiplied by -1)

    Args:
        g (nx.Graph): the graph to augment"""
    
    g = random_graph_affine_transformation(g, app_point='OSTIUM', vect=[1,0,0], alpha_rad=np.pi)

    return g

def trim_graph_random_branch(g: nx.Graph) -> nx.Graph:
    """Trims a random branch of the graph (i.e. deletes all the nodes that follow a random node which has a degree greater than 2,
    that represents a biforcation of the graph)

    Args:
        g (nx.Graph): the graph to augment"""
    #select a random node with degree greater than 2
    try:
        node = random.choice([n for n in g.nodes if g.degree[n] > 2])
    except:
        warnings.warn("The graph has no nodes with degree greater than 2")
        #hcatnetwork.draw.draw_simple_centerlines_graph_2d(g, backend="networkx")
        return g
    #check if the node attribute topology is 'INTERSECTION'
    try:
        assert g.nodes[node]['topology'].name == 'INTERSECTION', "The node selected is not an intersection"
    except:
        assert g.nodes[node]['topology'] == 'INTERSECTION', "The node selected is not an intersection"

    
    paths=find_connected_nodes(g, node)
    #discard the path that contains the 'OSTIUM' node
    #find the name of the ostium node that has as ['topology'] attribute 'OSTIUM'
    ostium_name = None
    for i, (name, feat_dict) in enumerate(g.nodes(data=True)):
        try:
            if feat_dict['topology'].name == 'OSTIUM': 
                ostium_name = name
                break
        except:
            if feat_dict['topology'] == 'OSTIUM':
                ostium_name = name
                break
    if ostium_name is None:
        warnings.warn("The ostium node has not been found, the graph may not have an ostium")

    paths = [path for path in paths if ostium_name not in path]
    #select as target_path the shortest path
    target_path = min(paths, key=len)
    #get a list of all the nodes that are not in the target_path (except the first node of the target_path)
    nodes_to_keep = [n for n in g.nodes if n not in target_path[1:]]
    #create a subgraph containing only the nodes in nodes_to_keep
    subgraph = get_subgraph_from_nbunch(g, nodes_to_keep)

    return subgraph

def add_graph_random_branch(g: nx.Graph) -> nx.Graph:
    """Adds a random branch to the graph (i.e. adds a random number of nodes and edges to the graph)

    Args:
        g (nx.Graph): the graph to augment"""
    
    #select a random node with degree equal to 2 and far at least 15 nodes from the closest endpoint
    nodes = [n for n in g.nodes if g.degree[n] == 2]
    nodes = [n for n in nodes if len(nx.shortest_path(g, n, target=None)) > 15] #the node must be far at least 15 nodes from the closest endpoint
    node = random.choice(nodes)
    #check if the node attribute topology is 'SEGMENT'
    try:
        assert g.nodes[node]['topology'].name == 'SEGMENT', "The node selected is not a segment"
    except:
        assert g.nodes[node]['topology'] == 'SEGMENT', "The node selected is not a segment"

    xyz_start = np.array([g.nodes[node]['x'], g.nodes[node]['y'], g.nodes[node]['z']], dtype=np.float32)
    r_start = g.nodes[node]['r'] #the radius at the given node
    side_start = g.nodes[node]['side'] #the side at the given node
    
    endpoints = [n for n in g.nodes if g.degree[n] == 1]
    endpoint = random.choice(endpoints)

    r_endpoint = g.nodes[endpoint]['r'] #the radius at the endpoint

    #find minimal distance between nodes in the graph edges
    min_dist_nodes=1000
    for i, (u, v, feat_dict) in enumerate(g.edges(data=True)):
        weight = g[u][v]['weight']
        if weight < min_dist_nodes:
            min_dist_nodes=weight

    #find, on the surface of a sferical coordinates neighborhood of the node, of radius equal to 10 times the min_dist, a point that is the most far apart from onther nodes of the graph
    #the point is the starting point of the new branch

    max_dist = 45*min_dist_nodes
    #create a list of all the nodes coordinates that are at maximum at a distance of max_dist from the node of xyz_start coordinates
    xyz_points = list()
    for i, (name, feat_dict) in enumerate(g.nodes(data=True)):
        xyz_coord=np.array([feat_dict['x'], feat_dict['y'], feat_dict['z']], dtype=np.float32)
        if np.linalg.norm(xyz_coord-xyz_start) <= max_dist:
            xyz_points.append(xyz_coord)
    #generate a list of 100 points evenly distributed on the surface of a sphere of radius max_dist centered in xyz_start
    xyz_points_on_sphere = generate_points_on_sphere(xyz_start, max_dist, 100)

    #select the point of xyz_points_on_sphere that is the most far apart from the other nodes of the graph in the list xyz_points
    max_dist=0
    for xyz_point in xyz_points_on_sphere:
        dist=0
        for xyz_point_2 in xyz_points:
            dist += np.linalg.norm(xyz_point-xyz_point_2)
        if dist > max_dist:
            max_dist=dist
            xyz_end = xyz_point

    #create the vessel
    vessel =  generate_fake_vessel(xyz_start, xyz_end, min_dist_nodes)
    #get ridd of the first node of the vessel since it is the same as the node of the graph that has been selected
    vessel = vessel[1:]
    num_nodes = len(vessel)
    #generate a list of radiuses for the nodes of the vessel from r_start to r_endpoint in a decreasing way
    r_vessel = np.linspace(r_start, r_endpoint, num_nodes)
    # xyz_graph_points = np.zeros((len(g.nodes), 3), dtype=np.float32)
    # for i, (name, feat_dict) in enumerate(g.nodes(data=True)):
    #     xyz_graph_points[i, 0] = float(feat_dict['x'])
    #     xyz_graph_points[i, 1] = float(feat_dict['y'])
    #     xyz_graph_points[i, 2] = float(feat_dict['z'])

    # #plot the points of the graph in a 3D plot
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(xyz_graph_points[:,0], xyz_graph_points[:,1], xyz_graph_points[:,2])
    # ax.scatter(xyz_start[0], xyz_start[1], xyz_start[2], c='r')
    # ax.scatter(xyz_end[0], xyz_end[1], xyz_end[2], c='r')
    # plt.show()
    #add the nodes and edges of the vessel to the graph
    last_node_name = node
    for i, xyz_coord in enumerate(vessel):
        index=0
        while True:
            new_node_name = str(len(g.nodes)+index)
            if new_node_name not in g.nodes:
                break
            else:
                index+=1

        g.add_node(new_node_name, x=float(xyz_coord[0]), y=float(xyz_coord[1]), z=float(xyz_coord[2]), r=r_vessel[i], topology=ArteryNodeTopology.SEGMENT, t=float(0), side=side_start)
        #find euclidean distance between last_node_name and new_node_name
        u_xyz_coord=np.array([g.nodes[last_node_name]['x'], g.nodes[last_node_name]['y'], g.nodes[last_node_name]['z']], dtype=np.float32)
        v_xyz_coord=np.array([g.nodes[new_node_name]['x'], g.nodes[new_node_name]['y'], g.nodes[new_node_name]['z']], dtype=np.float32)
        dist = float(np.linalg.norm(u_xyz_coord-v_xyz_coord))
        
        g.add_edge(new_node_name, last_node_name, weight=dist, euclidean_distance=dist)
        last_node_name = new_node_name

    #change the topology of the last node of the added vessel
    g.nodes[last_node_name]['topology'] = ArteryNodeTopology.ENDPOINT
    #plot all points of the graph in 3D
    # xyz_graph_points = np.zeros((len(g.nodes), 3), dtype=np.float32)
    # for i, (name, feat_dict) in enumerate(g.nodes(data=True)):
    #     xyz_graph_points[i, 0] = float(feat_dict['x'])
    #     xyz_graph_points[i, 1] = float(feat_dict['y'])
    #     xyz_graph_points[i, 2] = float(feat_dict['z'])

    # #plot the points of the graph in a 3D plot
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(xyz_graph_points[:,0], xyz_graph_points[:,1], xyz_graph_points[:,2])
    # ax.scatter(xyz_start[0], xyz_start[1], xyz_start[2], c='r')
    # ax.scatter(xyz_end[0], xyz_end[1], xyz_end[2], c='r')
    # plt.show()

    return g

if __name__ == "__main__":
    # Set up the dataset
    #dataset = ArteryGraphDataset(root='/home/erikfer/GNN_project/DATA/SPLITTED_ARTERIES_DATA/', ann_file='graphs_annotation.json')
    # Split the dataset into training and test sets with 80-20 splitting
    #set seeds for reproducibility
    seed = 42
    np.random.seed(seed)
    random.seed(seed)

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
        # hcatnetwork.draw.draw_simple_centerlines_graph_2d(g, backend="networkx")
        
        # aug_graph1 = trim_graph_random_branch(g)
        # hcatnetwork.draw.draw_simple_centerlines_graph_2d(aug_graph1, backend="networkx")
        # #hcatnetwork.draw.draw_simple_centerlines_graph_2d(aug_graph, backend="networkx")

        # aug_graph2= random_graph_portion_selection(g, 'random', 'OSTIUM')
        # hcatnetwork.draw.draw_simple_centerlines_graph_2d(aug_graph2, backend="networkx")

        # aug_graph3 = random_noise_on_node_position(g)
        # hcatnetwork.draw.draw_simple_centerlines_graph_2d(aug_graph3, backend="networkx")

        # graph = hcatnetwork.io.load_graph(file_path=os.path.join(folder, file_name),
        #                               output_type=hcatnetwork.graph.SimpleCenterlineGraph)

        # aug_graph4 = random_graph_affine_transformation(graph, 'OSTIUM')
        # hcatnetwork.draw.draw_simple_centerlines_graph_2d(aug_graph4, backend="networkx")
        # # aug_graph = random_node_addition(aug_graph, 1)
        # graph = hcatnetwork.io.load_graph(file_path=os.path.join(folder, file_name),
        #                               output_type=hcatnetwork.graph.SimpleCenterlineGraph)

        #hcatnetwork.draw.draw_simple_centerlines_graph_2d(g, backend="networkx")
        aug_graph5 = add_graph_random_branch(g)
        hcatnetwork.draw.draw_simple_centerlines_graph_2d(aug_graph5, backend="networkx")
    #print(max(max_dist_from_ostium), min(max_dist_from_ostium))