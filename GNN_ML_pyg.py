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
        # embedding layer
        x = F.relu(self.pre(x))
        # convolution layers
        x = F.relu(self.conv1(x, edge_index, edge_attr=edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr=edge_attr))
        # pool layer
        x = global_mean_pool(x, batch)
        # prediction layer
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x
        
        
        
script_dir=os.getcwd()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset=torch.load('polycry.pt')
dataset_training=dataset[400:]
dataset_test=dataset[:400]
torch.save(dataset_training, "polycry_train.pt")
torch.save(dataset_test, "polycry_test.pt")
dataset=torch.load('polycry_train.pt',map_location=device)
dataset_test=torch.load('polycry_test.pt',map_location=device)
node_n1=3
node_n2=10
node_n3=10
edge_n1=3
target_n1=3
model = GNNPOLY(node_n1, node_n2, node_n3, edge_n1,target_n1)

model.to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=0.002)
mse_cost_function = torch.nn.MSELoss()
iterations=400
BATCH_SIZE=40
BATCH_SIZE_test=20
my_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
my_loader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE_test, shuffle=True)
ep_num0=[]
loss_training=[]
loss_test=[]
for epoch in range(iterations):

        loss_temp=0
        iii0=0
        for dataset in my_loader:
            optimizer.zero_grad()            
            y_p=model(dataset)
            y_t=dataset.y
            mse0=mse_cost_function(y_p, y_t)        
            loss = mse0         
            loss.backward()
            optimizer.step()
            loss_temp=loss_temp+loss.item()
            iii0=iii0+1
        ep_num0.append(epoch)        
        loss_training.append(loss_temp/(iii0*1.0))        
        model.eval()
        loss_temp=0
        iii0=0
        for dataset_test in my_loader_test:
            y_p=model(dataset_test)
            y_t=dataset_test.y
            mse0=mse_cost_function(y_p, y_t)        
            loss = mse0
            loss_temp=loss_temp+loss.item()
            iii0=iii0+1
        loss_test.append(loss_temp/(iii0*1.0))
    
        rel_path = "models/GNN"+str(epoch+1)+".pt"
        abs_file_path = os.path.join(script_dir, rel_path)
        torch.save(model.state_dict(), abs_file_path)
loss1_training=np.array(loss_training).reshape(-1,1)  
np.savetxt("loss_training.csv", loss1_training, delimiter=",")
loss1_test=np.array(loss_test).reshape(-1,1)  
np.savetxt("loss_test.csv", loss1_test, delimiter=",")
torch.save(model.state_dict(), "GNN0.pt")
