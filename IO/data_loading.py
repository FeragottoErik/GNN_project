import os
import matplotlib 
import matplotlib.pyplot as plt
import json

import hcatnetwork
from HearticDatasetManager.asoca.dataset import DATASET_ASOCA_GRAPHS_RESAMPLED_05MM_DICT

# Path to the ASOCA dataset
ASOCA_PATH = '/home/erikfer/GNN_project/DATA/ASOCA/'
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
output_file_path = os.path.join(ASOCA_PATH, 'graph_annotation.json')
with open(output_file_path, "w") as json_file:
    json.dump(graph_annotation, json_file, indent=4)


for i, sample in enumerate(DATASET_ASOCA_GRAPHS_RESAMPLED_05MM_DICT["Normal"]): # "Normal", or "Diseased", from 0 to 19
    # Loading graph of ASOCA Dataset
    asoca_graph_file = os.path.join(
        ASOCA_PATH,
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

    left_ostium, right_ostium = g.get_coronary_ostia_node_id() #(left,right)
    left_artery_node_list= list()
    right_artery_node_list= list()

    for k, n in enumerate(g.nodes): # n is the node id as string
        node_left_ostium, node_right_ostium = g.get_relative_coronary_ostia_node_id(n)
        if node_left_ostium is not None: #the node is connected to the left ostium 
            #since no sample represent a codominant heart all nodes are either connected to right or left ostium
            left_artery_node_list.append(n)
        else:
            right_artery_node_list.append(n)

        #TODO: it may be computationally cheaper to use get_path_to_ostium() instead of get_relative_coronary_ostia_node_id()

    left_artery_graph = g.subgraph(left_artery_node_list)
    right_artery_graph = g.subgraph(right_artery_node_list)

    if VERBOSE:
        print("{} \n left artery: {} nodes \n right artery {} nodes, \n whole graph {} nodes".format(g.graph["image_id"], len(left_artery_node_list), len(right_artery_node_list), k+1))

    if PLOT:
        hcatnetwork.draw.draw_simple_centerlines_graph_2d(left_artery_graph, backend="networkx")
        hcatnetwork.draw.draw_simple_centerlines_graph_2d(right_artery_graph, backend="networkx")

    hcatnetwork.io.save_graph(left_artery_graph, os.path.join(ASOCA_PATH, "left_artery", g.graph["image_id"]+"_left.gml"))
    hcatnetwork.io.save_graph(right_artery_graph, os.path.join(ASOCA_PATH, "right_artery", g.graph["image_id"]+"_right.gml"))

    additional_graphs = [
        {
            "id": 2*i, 
            "file_name": os.path.join("left_artery", g.graph["image_id"]+"_left.gml"),
            "ostium_id": left_ostium,
            "category_id": 1
        },
        {
            "id": 2*i+1,
            "file_name": os.path.join("right_artery", g.graph["image_id"]+"_right.gml"),
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




    

