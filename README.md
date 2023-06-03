# Neural Sculpting: Uncovering hierarchically modular task structure through pruning and network analysis


This repository is the official implementation of the paper: https://arxiv.org/abs/2305.18402

Example hierarchically modular task and uncovered structure through Neural Sculpting


https://github.com/ShreyasMalakarjunPatil/Neural-Sculpting/assets/56805727/0db95811-503b-45da-b15a-29e140186a0d



To install requirements run:

```setup
pip3 install -r requirements.txt
```

## Code Details 

A detailed description of the code base is provided below. 
The code is divided into 4 directories:

### run

The directory cosists of 3 files to run training. 

* The file train.py simply runs the training of a dense NN and saves the model. 
An example command to run: 

```
python3 main.py --experiment train --task boolean --dataset_path ./datasets/exemplar/ --result_path ./results/exemplar/dense/ --modularity 4 2 4 --arch 4 24 4 --lr 0.05 --batch_size 8 --epochs 15 --seed 0 
```
* The file prune.py runs the unit pruning algorithm. First the dense NN is trained with the given hyper-parameters. Then hidden units are pruned iteratively (with re-training) until the sparse NN can no longer achieve the validation accuracy specified. 

An example command to run:
```
python3 main.py --experiment prune --task boolean --dataset_path ./datasets/exemplar/ --result_path ./results/exemplar/unit_edge/ --modularity 4 2 4 --arch 4 24 4 --lr 0.05 --batch_size 8 --epochs 15 --seed 0 --prune_units 1 --unit_pruning_gradient 70.0 --prune_acc_threshold 100.0
```
To run the edge pruning after unit pruning:
```
python3 main.py --experiment prune --task boolean --dataset_path ./datasets/exemplar/ --result_path ./results/exemplar/unit_edge/ --modularity 4 2 4 --arch 4 24 4 --lr 0.05 --batch_size 8 --epochs 15 --seed 0 --prune_units 0 --unit_pruning_gradient 70.0 --prune_acc_threshold 100.0
```
 
### detection_pipeline

The directory detection_pipeline consist of different components used in the module detection method proposed in the paper.

* cluster.py -> runs the clustering algorithms and fins the optimal number of clusters on the basis of modularity metric and unit separability tests.
* merge_clusters.py -> takes as input the clusters obtained and merges them on the basis of merging_threshold provided through arguments
* visualize.py -> considers the final set of modules and plots visualizations by rearanging the unit positions on the basis of their modules and layers
* utils -> consists of various utility functions including modularity metric computation, adjacency matrix computation, clustering eature vector computation and unit separability tests

The commands to run module detection and visualize the NNs along with the hyper-parameters are detailed in the directory "shell".

An example command to run module detection and visualize the resulting hierarchical modular structure: 
```
python3 main.py --experiment module_detection --task boolean --result_path ./results/exemplar/unit_edge/ --modularity 4 2 4 --arch 4 24 4 --lr 0.05 --batch_size 8 --epochs 15 --seed 0 --unit_pruning_gradient 70.0 --pruning_gradient 2.5 --feature_step_size 2 --clusterability_step_size 2 --visualization_path ./visualizations/exemplar/
```

### models

The directory models consists of the mlp architectures implemented for each NN deph, and the implementation of masked linear layers in layers.layers.py

### utils

The directory utils consists of 5 files:

* load.py -> this script is used across experiments and runs to load models, datasets, optimizers, loss functions etc. on the basis of the arguments input to the parser
* train.py -> runs the training iterations across all experiments
* loss.py -> implements the loss functions used
* prune_units.py -> implements the scoring mechanism for unit pruning along with generating the weight and bias masks.
* prune.py -> implements the scoring mechanism and the masks for edge pruning

## Other arguments and hyper-parameters

Find a full list of hyper-parameters and values through:
```
python3 main.py --help
```

## Result details, Hyper-parameters and Visualizations

Please find the visualizations for NNs trianed on all the function graphs in the directory named "visualizations". 
The directory consists of 7 additional folders that include 5 NN visualization folders, 1 folder with Notebooks and 1 folder with success rate plots. 
The 5 NN visualization folders are for one experiment each. 

* exemplar -> four function graphs used for hyper-parameter tuning and validation of the proposed pipeline

* non_separable_increasing_reuse -> function graphs with a single reused sub-function and increasing use

* input_overlap_exp -> function graphs with two input separable sub-functions and increasing overlap between the two input sets

* increasing_hierarchical_levels -> functions graphs with multiple hierarchical levels used in the experiment to show that different NN depth may result in different hierarchical structure recovered

Folders with the same names can also be found in directories "results", "shell" and "datasets. 
The directory "results" consists of the trained models and masks. 
The directory "shell" consists of commands to run module detection on those models and generate the NN visualizations.

For detailed list of hyper-parameters used for each of the experiments please refer to the directory "shell" and the commands.
