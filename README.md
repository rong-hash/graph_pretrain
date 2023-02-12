# When to Pre-Train Graph Neural Networks? An Answer from Data Generation Perspective!
## About

This project is the implementation of paper “When to Pre-Train Graph Neural Networks? An Answer from Data Generation Perspective!"

## Dependencies
The script has been tested running under Python 3.6.5, with the following packages installed (along with their dependencies):

pytorch                   1.6.0+cu101 

torch-geometric           1.6.3        

rdkit                     2020.09.1.0 

networkx                  2.3

scikit-learn              0.20.3 

## File Folders
node classification: contains the code of estimating graph pre-training feasibility and pre-traing&fine-tuning data for graph classification  

graph classification: contains the code of estimating graph pre-training feasibility and pre-traing&fine-tuning data for node classification  

utils:contains the code of models.

## Dataset Download
For the graph classiciation pre-training and fine-tuning dataset, download from [pre-training data](https://github.com/snap-stanford/pretrain-gnns#dataset-download), unzip it, and put it under graph_classification/data/dataset/

For the node classiciation pre-training dataset, the original datasets are stored in **data.bin**. And the datasets can be download through [pre-training data](https://drive.google.com/file/d/1kbOciSHXSOAFV7X1CuL_nm9_2sxKeDfU/view), unzip it, and put it under node_classification/data/dataset/

## Usage: How to run the code
To run the node_classification, see example below

`python node_classification/estimate_feasiblity.py --pre_data imdb_facebook --down_data h-index`

To run the graph_classification, see example below

`python graph_classification/estimate_feasiblity.py --pre_data zinc_standard_agent --split_num 2 --split_ids 01 --down_data bbbp `

