import torch
import torch.nn as nn
import torch.nn.functional as F

class GCN(nn.Module):

	def __init__(self , A, dim_in, dim_out, bias=True, drop=None, bn=False, init='default', agg_first=True, attn=False, part=False):
		super(GCN, self).__init__()
		self.A = A
		self.fc1 = nn.Linear(dim_in, dim_out, bias=bias)
		self._initialize_weights(mod=init)
		self.drop = drop
		self.bn = bn
		self.agg_first = agg_first
		self.attn = attn
		self.part = part
		if attn:
			self.m = (self.A > 0)
			self.M = nn.Parameter(torch.ones(1,len(self.m.nonzero()), dtype=torch.float))
		if drop:
			self.dropout = nn.Dropout(drop) 
		if bn:
			# (N,L,C) or (N, C)
			# self.batchnorm = nn.BatchNorm1d(dim_in)
			if part:	self.batchnorm = nn.BatchNorm1d(5)
			else:  self.batchnorm = nn.BatchNorm1d(17)


	def _initialize_weights(self, mod):
		'''
		TODO: add some common methods of initialization
		'''
		if mod == 'default':
			pass
		if mod == 'kaiming_uniform':
		    nn.init.kaiming_uniform_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
		    nn.init.constant_(self.fc1.bias, 0)
		if mod == 'xavier_uniform':
		    nn.init.xavier_uniform_(self.fc1.weight)
		    nn.init.constant_(self.fc1.bias, 0)

	def forward(self, X):
		'''
		Z = AXW or Z=A(XW)
		'agg_first': aggregation first, which mean Z=AXW
		!NOTE: A's size=[k_n,k_n] and X's size=[b_n,k_n,c_n], we can not calculate 'AX' directly 
		'''
		# import ipdb; ipdb.set_trace()
		self.A = self.A.to(X.device)
		b_n, k_n, c_n = X.shape
		if self.attn:
			adj = -9e15 * torch.ones_like(self.A).to(X.device)
			adj[self.m] = self.M
			self.A = nn.Softmax(dim=1)(adj)
		if self.agg_first:
			# print(1)
			X = X.permute(1,0,2).contiguous().view(k_n,-1)
			X = self.A.mm(X).view(k_n,b_n,-1).permute(1,0,2)
			if self.part:
				# import ipdb; ipdb.set_trace()
				X = X[:,[0,5,6,11,12],:]
			if self.bn:
				X = self.batchnorm(X)
			X = F.relu(self.fc1(X))
			if self.drop:
				return self.dropout(X)
			else:
				return X
		else:
			# print(1)
			X = self.fc1(X)
			X = X.permute(1,0,2).contiguous().view(k_n,-1)
			X = self.A.mm(X).view(k_n,b_n,-1).permute(1,0,2)
			# X = F.relu(torch.mm(self.A, X))
			if self.bn:
				X = self.batchnorm(X)
			X = F.relu(X)
			if self.drop:
				return self.dropout(X)
			else:
				return X