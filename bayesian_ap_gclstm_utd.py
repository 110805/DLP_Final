import numpy as np
import pickle
import torch
from torch.utils.data import Dataset
import sys
import torch.nn as nn
import torch.nn.functional as F
import time
import glob
import scipy.io
import os
import math

from blitz.modules import BayesianLinear, BayesianConv2d, BayesianLSTM
from blitz.utils import variational_estimator

num_joint = 20
max_frame = 125
input_feature = 6
num_feature = 16
hidden_size = 128
batch_size = 32
learning_rate = 0.001
momentum = 0.9
decay_rate = 0.9
decay_step = 100
epochs = 1000
device = 'cuda:0'
path = "UTD_AP/"

class UTDDataset(Dataset):
    def __init__(self, data_path):
        super(UTDDataset, self).__init__()
        self.data_path = data_path
        self.load_data()
        
    def load_data(self):
        path_pattern = self.data_path + '*.mat'
        files_list = glob.glob(path_pattern, recursive=True)
        self.data = torch.zeros((len(files_list),input_feature,max_frame,num_joint),dtype=torch.float32)
        self.labels = []
        self.num_frame = []
        for i,file_name in enumerate(files_list):
            a = os.path.basename(file_name).split('_')[0]
            mat = scipy.io.loadmat(file_name)['d_skel'].astype("float32")
            mat = mat.transpose((1,2,0)) # transpose to (C, #frame, #joint)
            mat -= np.expand_dims(mat[:,:,2],axis=2) # set center at spine of body
            
            aug_mat = np.concatenate((mat,mat),axis=0) # concat speed of xyz-axis
            aug_mat[3:,:,:] -= np.roll(mat,1,axis=1) # speed = x(t) - x(t-1)
            aug_mat[3:,0,:] = 0 # 1st frame, speed = 0
            
            #self.data.append(aug_mat)
            frame = aug_mat.shape[1]
            self.data[i,:,:frame] = torch.from_numpy(aug_mat)
            self.num_frame.append(frame)
            self.labels.append(int(a[1:])-1)
    
    def __getitem__(self, index):
        data = self.data[index]
        label = self.labels[index]
        f = self.num_frame[index]
        return data, label, f

    def __len__(self):
        return len(self.labels)
    
    def __iter__(self):
        return self
    
train_dataset = UTDDataset(data_path=path+'train/') # subject 1, 3, 5
valid_dataset = UTDDataset(data_path=path+'valid/') # subject 7
test_dataset = UTDDataset(data_path=path+'test/')   # subject 2, 4, 6, 8

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=len(test_dataset),shuffle=False)

"""
1. head; 
2. shoulder_center;
3. spine;
4. hip_center;
5. left_shoulder;
6. left_elbow;
7. left_wrist;
8. left_hand;
9. right_shoulder;
10. right_elbow;
11. right_wrist;
12. right_hand;
13. left_hip;
14. left_knee;
15. left_ankle;
16. left_foot;
17. right_hip;
18. right_knee;
19. right_ankle;
20. right_foot;
"""
inward_ori_index = [(1,2),(3,2),(4,3),(5,2),(6,5),(7,6),(8,7),(9,2),(10,9),(11,10),
                    (12,11),(13,4),(14,13),(15,14),(16,15),(17,4),(18,17),(19,18),(20,19)]
inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
five_key_point = [1,7,11,15,19]

def normalize(A):
    rowsum = torch.sum(A, 0)
    r_inv = torch.pow(rowsum, -0.5)
    r_mat_inv = torch.diag(r_inv).float()

    A_norm = torch.mm(r_mat_inv, A)
    A_norm = torch.mm(A_norm, r_mat_inv)

    return A_norm
    
def gen_adj():
    A = torch.zeros(3,num_joint,num_joint,dtype=torch.float)
    for (i,j) in inward:
        A[0,j,i] = 1
    for (i,j) in outward:
        A[1,j,i] = 1
    for i in five_key_point:
        for j in five_key_point:
            A[2,i,j] = 1
    for i in range(num_joint):
        A[:,i,i] = 1
    A[0] = normalize(A[0])
    A[1] = normalize(A[1])
    A[2] = normalize(A[2])
    return A

from torch.nn.parameter import Parameter
class GraphConvolution(nn.Module):
    def __init__(self, num_graph, in_feature, out_feature):
        super(GraphConvolution, self).__init__()
        self.num_graph = num_graph
        self.in_feature = in_feature
        self.out_feature = out_feature
        
        self.mask = nn.Parameter(torch.ones(num_graph, num_joint, num_joint))
        
        self.gcn_list = nn.ModuleList([
            BayesianConv2d(
                self.in_feature,
                self.out_feature,
                kernel_size=(1, 1)) for i in range(self.num_graph)
        ])
        
        self.bn = nn.BatchNorm2d(out_feature)
        self.act = nn.ReLU()

    def forward(self, adj, x):
        # x : B*f*T*20
        N, C, T, V = x.size()
        
        adj = adj * self.mask
        
        for i,a in enumerate(adj):
            xa = x.view(-1,V).mm(a).view(N,C,T,V)
            if i == 0:
                y = self.gcn_list[i](xa)
            else:
                y += self.gcn_list[i](xa)
                
        y = self.bn(y)
        
        return self.act(y)
    
class GCLayers(nn.Module):
    def __init__(self,num_feature,num_graph):
        super(GCLayers, self).__init__()
        self.num_feature = num_feature
        self.gc1 = GraphConvolution(num_graph,input_feature,num_feature)
        self.gc2 = GraphConvolution(num_graph,num_feature,num_feature)
        self.gc3 = GraphConvolution(num_graph,num_feature,num_feature)
        self.gc4 = GraphConvolution(num_graph,num_feature,num_feature)
        
    def forward(self, adj, x):
        # x : B*6*T*20
        output1 = self.gc1(adj,x)
        output2 = self.gc2(adj,output1)
        output3 = self.gc3(adj,output2) + output1
        output4 = self.gc4(adj,output3) + output2
        return output4

@variational_estimator
class GC_LSTM(nn.Module):
    def __init__(self, num_feature, hidden_size):
        super(GC_LSTM, self).__init__()
        self.adj = gen_adj().to(device)
        self.num_graph = self.adj.shape[0]
        self.num_feature = num_feature
        self.hidden_size = hidden_size
        self.output_feature = num_feature
        
        self.datat_bn = nn.BatchNorm1d(input_feature * num_joint)
        self.gclayers = GCLayers(num_feature,self.num_graph)
        self.dropout = nn.Dropout(0.25)
        #self.lstm = BayesianLSTM(self.output_feature*num_joint,hidden_size,prior_sigma_1=1,prior_pi=1,posterior_rho_init=-3.0)
        self.lstm = nn.LSTM(self.output_feature*num_joint,hidden_size)
        self.classifier = nn.Linear(hidden_size, 27)
        self.discriminator = nn.Linear(hidden_size, 1)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        
        
    def forward(self, x, num_frame):
        # x : B*6*T*20
        x = self.gclayers(self.adj,x)
        x = self.dropout(x)

        N,C,T,V = x.size()
        x = x.permute(0,2,1,3).contiguous().view(N,T,1,C*V)
        for i in range(N):
            if i == 0:
                output = self.lstm(x[i,:num_frame[i]])[0][-1] # 取lstm最後一個output
            else:
                output = torch.cat((output,self.lstm(x[i,:num_frame[i]])[0][-1]))
        
        output = self.dropout(output)
        out1 = self.classifier(output)
        out2 = self.discriminator(output)
        return self.softmax(out1), self.sigmoid(out2)
            
net = GC_LSTM(num_feature,hidden_size)
net = net.to(device)

CE_criterion = nn.CrossEntropyLoss()
BCE_criterion = nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=decay_step,gamma=decay_rate)

kl_weight = 1. / len(train_dataset)

M = 10 # for Monte Carlo estimation

test_interval = 10
early_stop = 0.92
training_Dloss = []
training_Gloss = []
start = time.time()
net.train()

validloader_iter = iter(valid_loader)

for epoch in range(epochs):
    G_LOSS = 0
    D_LOSS = 0
    print("{:3d} epoch".format(epoch+1),end=", ")
    correct = 0
    for i,(data, label, num_frame) in enumerate(train_loader):
        try:
            valid_data, valid_label, valid_f = next(validloader_iter)
        except StopIteration:
            validloader_iter = iter(valid_loader)
            valid_data, valid_label, valid_f = next(validloader_iter)
        
        data, label, num_frame = data.to(device), label.to(device), num_frame.to(device)
        valid_data, valid_label, valid_f = valid_data.to(device), valid_label.to(device), valid_f.to(device)
        
        positive = torch.ones(label.size()).to(device)
        valid_positive = torch.ones(valid_label.size()).to(device)
        valid_negative = torch.zeros(valid_label.size()).to(device)
        
        # train discriminator
        optimizer.zero_grad()
        for m in range(M):
            _, output = net(data,num_frame)
            output = output.squeeze()
            D_positive_loss = BCE_criterion(output,positive)
            
            _, valid_output = net(valid_data,valid_f)
            valid_output = valid_output.squeeze()
            D_negative_loss = BCE_criterion(valid_output,valid_negative)
            
            D_loss = (D_positive_loss + D_negative_loss) / M
            
            D_loss.backward()
            D_LOSS += D_loss.item()
        optimizer.step()
        
        # train GC-LSTM and Classifier
        optimizer.zero_grad()
        for m in range(M):
            output, _ = net(data,num_frame)
            class_loss = CE_criterion(output, label)
            
            _, valid_output = net(valid_data,valid_f)
            valid_output = valid_output.squeeze()
            adversarial_loss = BCE_criterion(valid_output,valid_positive)
            
            # G_kl_loss = net.nn_kl_divergence() * kl_weight
            
            G_loss = (class_loss + adversarial_loss) / M
            G_loss.backward()
            G_LOSS += G_loss.item()
            
            _, pred = output.max(1)
            correct += pred.eq(label).sum().item()
        optimizer.step()
    
    correct /= M
    print("D loss:{:6.4f}, G loss:{:6.4f}, training acc:{:6.2f}%, time:{:.2f}s"
          .format(D_LOSS/len(train_dataset), G_LOSS/len(train_dataset), correct/len(train_dataset)*100.,time.time()-start))

    scheduler.step()
    if (epoch+1) % test_interval == 0:
        correct = 0
        with torch.no_grad():
            for (data, label, num_frame) in test_loader:
                data, label, num_frame = data.to(device), label.to(device), num_frame.to(device)
                for m in range(M):
                    output, _ = net(data,num_frame)
                    _, pred = output.max(1)
                    correct += pred.eq(label).sum().item()

        correct /= M
        print("test acc: {:5.2f}%, time:{:7.2f}s"
              .format(correct/len(test_dataset)*100.,time.time()-start))
        if correct/len(test_dataset) > early_stop:
            torch.save(net,"Bayesian_GC_LSTM.pkl")
            break
