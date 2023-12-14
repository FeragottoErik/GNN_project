# Coronary_arteries_classification

## Description

Given a dataset composed by graph extracted from centerlines of coronary arteries (thanks to HCATNetwork package), the package allows to:
-split the graph into two subgraphs of left and right CA, normlize the coordinates between 0 and 1 on xyz and center the graph on the ostium (0,0,0)
-generate a .JSON file with graph level annotation for further graph classification
-transform the networkx .GML graphs into PyGeo data
-create a PytorchGeometrics dataset object to facilitate graph loading and processing during model training and testing
-augmentations on graphs that encompass branch trimming, graph portion selection, random graph branch addition, graph flipping, random noise on node position etc.
-Utilize a 3-layer GIN composed by 3 GINconv layers, a pooling layer and a FC layer that lead to a binary class output 
-Utilize a 3-layer GAT network as above
-Custom functions that allow activation visualization and ease of implementation of further functions

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/FeragottoErik/GNN_project.git
    ```

2. Navigate to the project directory:

    ```bash
    cd project-directory
    ```

3. Install the dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

First of all create the processed dataset with splitted RCA-LCA and JSON file by means of preprocessing/right_left_splitting.py
Run main.py once, it will automatically process the data to create a new 'processed' folder in which .pt files are saved for each graph.
It will then throw an error due to misimplementation of dataset.py
create 'ref_data_list.txt' in 'processed' parent folder containing a list of all elements included in 'processed' folder
Run again the main.py to train test and evaluate the model.

