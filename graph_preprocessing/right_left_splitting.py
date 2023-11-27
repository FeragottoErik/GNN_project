import os
import matplotlib 
import matplotlib.pyplot as plt
import json
from warnings import warn
import networkx as nx

import hcatnetwork
from HearticDatasetManager.asoca.dataset import DATASET_ASOCA_GRAPHS_RESAMPLED_05MM_DICT
from HearticDatasetManager.cat08.dataset import DATASET_CAT08_GRAPHS

from Utilities.custom_functions import get_subgraph_from_nbunch 

ASOCA_PATH = '/home/erikfer/GNN_project/DATA/ASOCA/'
CAT08_PATH = '/home/erikfer/GNN_project/DATA/CAT08/'
DATA_PATH = '/home/erikfer/GNN_project/DATA/'
VERBOSE = True
PLOT = False


graph_annotation = {
    "graphs": [
        #id, file_name, ostium_id, category_id
        #add graph entries as needed
    ],
    "categories": [
        {
            "id": 1,
            "name": "left",
            "suercategory": "coronary_artery"
        },
        {
            "id": 2,
            "name": "right",
             "suercategory": "coronary_artery"
        },
    ]
}

# Save to a JSON file
output_file_path = os.path.join(DATA_PATH, 'graphs_annotation.json')

if os.path.exists(output_file_path):
    # Check if the file is not empty
    if os.path.getsize(output_file_path) > 0:
        # File exists and is not empty
        with open(output_file_path, "r") as json_file:
            data = json.load(json_file)
            if VERBOSE:
                print("JSON file exists and is not empty. Data:", data)
            start_id = len(data['graphs'])
        # File exists but is empty
    else:
        if VERBOSE:
            print("JSON file exists but is empty.")
else:
    start_id=0
    with open(output_file_path, "w") as json_file:
        json.dump(graph_annotation, json_file, indent=4)


for i, sample in enumerate(DATASET_CAT08_GRAPHS): # "Normal", or "Diseased" for ASOCA, nothing for CAT08
    # Loading graph of ASOCA Dataset
    asoca_graph_file = os.path.join(
        CAT08_PATH,
        sample
    )

    g = hcatnetwork.io.load_graph(
        file_path=asoca_graph_file,
        output_type=hcatnetwork.graph.SimpleCenterlineGraph
    )
    
    if VERBOSE:
        print("{} - Processing graph: {}".format(i, g.graph["image_id"]))


    # Plotting the graph
    if PLOT:
        hcatnetwork.draw.draw_simple_centerlines_graph_2d(g, backend="networkx")

     
    left_ostium, right_ostium = tuple(g.get_coronary_ostia_node_id()) #(left,right)
    left_artery_node_list= list()
    right_artery_node_list= list()

    for k, n in enumerate(g.nodes): # n is the node id as string

        if len(g.get_relative_coronary_ostia_node_id(n)) > 1: #to spot codominant hearts
            warn("Warning: node {} is connected to more than one ostium".format(n))

        node_ostium=g.get_relative_coronary_ostia_node_id(n)[0] #first element of the tuple is the ostium id
        if node_ostium == left_ostium: #the node is connected to the left ostium 
            #since no sample represent a codominant heart all nodes are either connected to right or left ostium
            left_artery_node_list.append(n)
        else:
            right_artery_node_list.append(n)

        #TODO: it may be computationally cheaper to use get_path_to_ostium() instead of get_relative_coronary_ostia_node_id()

    left_artery_graph = get_subgraph_from_nbunch(g, left_artery_node_list)
    right_artery_graph = get_subgraph_from_nbunch(g, right_artery_node_list)

    if VERBOSE:
        print("{} \n left artery: {} nodes \n right artery {} nodes, \n whole graph {} nodes".format(g.graph["image_id"].split(' ')[0], len(left_artery_node_list), len(right_artery_node_list), k+1))

    if PLOT:
        hcatnetwork.draw.draw_simple_centerlines_graph_2d(left_artery_graph, backend="networkx")
        hcatnetwork.draw.draw_simple_centerlines_graph_2d(right_artery_graph, backend="networkx")

    hcatnetwork.io.save_graph(left_artery_graph, os.path.join(DATA_PATH, "left_artery", left_artery_graph.graph["image_id"].split(' ')[0]+"_left.gml"))
    hcatnetwork.io.save_graph(right_artery_graph, os.path.join(DATA_PATH, "right_artery", right_artery_graph.graph["image_id"].split(' ')[0]+"_right.gml"))

    additional_graphs = [
        {
            "id": start_id+2*i, 
            "file_name": os.path.join("left_artery", g.graph["image_id"].split(' ')[0]+"_left.gml"),
            "ostium_id": left_ostium,
            "category_id": 1
        },
        {
            "id": start_id+2*i+1,
            "file_name": os.path.join("right_artery", g.graph["image_id"].split(' ')[0]+"_right.gml"),
            "ostium_id": right_ostium,
            "category_id": 2
        }
    ]

    # Add more graph entries as needed

    with open(output_file_path, "r+") as json_file:
        # Load existing data
        existing_data = json.load(json_file)

        # Add new items to the 'graphs' list
        existing_data["graphs"].extend(additional_graphs)

        # Move the file cursor to the beginning for overwriting
        json_file.seek(0)

        # Write the updated data back to the file
        json.dump(existing_data, json_file, indent=4)




    

