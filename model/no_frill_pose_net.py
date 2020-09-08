import torch
import torch.nn as nn
import torch.nn.functional as F

class fully_connect(nn.Module):
    def __init__(self, action_num, drop):
        super(fully_connect, self).__init__()
        self.linear1 = nn.Linear(2*17*2, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, action_num)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.drop = drop
        if drop:
            self.dropout = nn.Dropout(drop)
    
    def forward(self, x1, x2):
        # import ipdb; ipdb.set_trace()
        x = torch.cat((x1.view(x1.shape[0],-1), x2.view(x2.shape[0],-1)), dim=1)
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.drop:
            x = self.dropout(x)

        x = self.linear2(x)
        x = self.bn2(x)
        x = self.relu(x)
        if self.drop:
            x = self.dropout(x)

        x = self.linear3(x)
        
        return x