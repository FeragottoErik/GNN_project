import networkx as nx
from hcatnetwork.graph import SimpleCenterlineGraph
import hcatnetwork

def get_subgraph_from_nbunch(g: SimpleCenterlineGraph, node_list: list) -> nx.classes.graph.Graph:
    
    """Custum function to generate a graph of class SimpleCenterlineGraph that is the subgraph deriving from a given list of nodes"""
    SG=g.__class__(**g.graph) #setting the graph-level attributes copied from original graph
    SG.graph["image_id"] += " - subgraph" #changeing the name at graph level
    SG.add_nodes_from((n, g.nodes[n]) for n in node_list)
    SG.add_edges_from((n, nbr, d)
                    for n, nbrs in g.adj.items() if n in node_list
                    for nbr, d in nbrs.items() if nbr in node_list)
    SG.graph.update(g.graph)
    return SG