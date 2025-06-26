from __future__ import print_function, division
import argparse
import os
import shutil
import sys
import time
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from torch_geometric.data import Data
import math

from torch_geometric.nn import Linear, CGConv, global_mean_pool

class GNNPOLY(torch.nn.Module):
    '''
    Graph Neural Network for Polycrystalline materials
    '''
    #node_n1: original node feature number
    #node_n2: node feature number after preprocessing
    #node_n3: node feature number after convolution
    #edge_n1: edge feature number
    #target_n1: number of targets
    def __init__(self, node_n1, node_n2, node_n3, edge_n1,target_n1):
        super(GNNPOLY, self).__init__()
        self.pre = Linear(node_n1, node_n2)
        self.conv1 = CGConv(node_n2, dim=edge_n1)
        self.conv2 = CGConv(node_n2, dim=edge_n1)
        self.fc1 = Linear(node_n3, 128)
        self.fc2= Linear(128, 128)
        self.out = Linear(128, target_n1)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        # Pre Processing Linear Layer
        x = F.relu(self.pre(x))
        # 1. Obtain node embeddings
        x = F.relu(self.conv1(x, edge_index, edge_attr=edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr=edge_attr))
        # 2. Readout layer
        x = global_mean_pool(x, batch)
        # 3. Apply Fully Connected Layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x
        
        
        

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
node_n1=3
node_n2=10
node_n3=10
edge_n1=3
target_n1=3

model0=GNNPOLY(node_n1, node_n2, node_n3, edge_n1,target_n1)
model0.load_state_dict(torch.load("GNN0.pt",map_location=torch.device('cpu')))
model0.eval()
dataset=torch.load('polycry_test.pt',map_location=torch.device('cpu'))
BATCH_SIZE=1000
my_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
dataset_size = len(my_loader.dataset)

for dataset in my_loader:
    pre0=model0(dataset)
    pre1=pre0.data.cpu().numpy()
    np.savetxt("pre1.csv", pre1, delimiter=",")
    targetlist0=dataset.y.data.cpu().numpy()
    np.savetxt("target.csv", targetlist0, delimiter=",")
# ep_num1=np.array(ep_num0).reshape(-1,1)
# loss1=loss1_training

# import matplotlib as mpl
# import matplotlib.pyplot as plt
# import numpy as np


# fig, ax = plt.subplots(figsize=(4, 3))

# ax.plot(ep_num1, loss1, label='loss')  # Plot some data on the axes.
# ax.set_xlabel('epoch number')  # Add an x-label to the axes.
# ax.set_ylabel('loss')  # Add a y-label to the axes.
# #ax.set_title("Simple Plot")  # Add a title to the axes.
# ax.legend();  # Add a legend.
# plt.show()

# model.eval()
# pre0=model(nfeature,neighblist,efeature)
# print(pre0)
