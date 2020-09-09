import torch
import torch.nn as nn
import torch.nn.functional as F
from model.gcn import GCN
import numpy as np

def adj_construction(inds, keypoint_num=17, symmetric=True):
	# # A = A+I
	# A = A + torch.eye(A.size(0))
	if inds:
		A = torch.zeros((keypoint_num, keypoint_num))
		for i, j in inds.items():
			A[i,j] = 1
	else:
		A = torch.ones((keypoint_num,keypoint_num))
	# D
	d = A.sum(1)
	if symmetric:
		#D = D^-1/2
		D = torch.diag(torch.pow(d , -0.5))
		return D.mm(A).mm(D)
	else :
		# D=D^-1
		D =torch.diag(torch.pow(d,-1))
		return D.mm(A)

class PGception_Layer(nn.Module):	

	def __init__(self, in_channel, out_channel_list, branch_list=[0,1,2,3], bias=True, drop=None, bn=False, agg_first=True, attn=False, init='kaiming_uniform'): # init = 'default', 'kaiming_uniform', 'xavier_uniform'
		super(PGception_Layer, self).__init__()
		# prepare adjacent matrixs
		keypoint_num = 17
		adj0 = torch.eye(keypoint_num)
		adj1_inds = {0:[0,1,2,5,6], 1:[0,1,3], 2:[0,2,4], 3:[1,3], 4:[2,4], 5:[0,5,6,7,11], 6:[0,5,6,8,12], 7:[5,7,9], 8:[6,8,10], 
					9:[7,9], 10:[8,10], 11:[5,11,12,13], 12:[6,11,12,14], 13:[11,13,15], 14:[12,14,16], 15:[13,15], 16:[14,16]}
		adj1 = adj_construction(inds=adj1_inds, keypoint_num=keypoint_num, symmetric=True)
		adj2_inds = {0:[0,1,2,3,4,5,6,7,8,11,12], 1:[0,1,2,3,5,6], 2:[0,1,2,4,5,6], 3:[0,1,3], 4:[0,2,4], 5:[0,1,2,5,6,7,8,9,11,12,13], 6:[0,1,2,5,6,7,8,10,11,12,14], 7:[0,5,6,7,9,11], 8:[0,5,6,8,10,12], 
					9:[5,7,9], 10:[6,8,10], 11:[0,5,6,7,11,12,13,14,15], 12:[0,5,6,8,11,12,13,14,16], 13:[5,11,12,13,15], 14:[6,11,12,14,16], 15:[11,13,15], 16:[12,14,16]}
		adj2 = adj_construction(inds=adj2_inds, keypoint_num=keypoint_num, symmetric=True)
		adj_all = adj_construction(inds=None)

		adj_part_inds = {0:[0,1,2,3,4], 1:[1], 2:[2], 3:[3], 4:[4], 5:[5,7,9], 6:[6,8,10], 7:[7], 8:[8], 9:[9], 10:[10], 11:[11,13,15], 12:[12,14,16], 13:[13], 14:[14], 15:[15], 16:[16]}
		adj_part = adj_construction(inds=adj_part_inds, keypoint_num=keypoint_num, symmetric=True)
		A = [adj0, adj1, adj2, adj_all, adj_part]
		
		# import ipdb; ipdb.set_trace()
		self.branch_list = branch_list
		if 0 in self.branch_list:
			self.branch_0 = GCN(A[0], in_channel, out_channel_list[0], bias=bias, drop=drop, bn=bn, init=init, agg_first=agg_first)
		if 1 in self.branch_list:
			self.branch_1 = GCN(A[1], in_channel, out_channel_list[1], bias=bias, drop=drop, bn=bn, init=init, agg_first=agg_first, attn=attn)
		if 2 in self.branch_list:
			self.branch_2 = GCN(A[2], in_channel, out_channel_list[2], bias=bias, drop=drop, bn=bn, init=init, agg_first=agg_first, attn=attn)
		if 3 in self.branch_list:
			self.branch_all = GCN(A[3], in_channel, out_channel_list[3], bias=bias, drop=drop, bn=bn, init=init, agg_first=agg_first, attn=attn)
		if 4 in self.branch_list:
			self.branch_part = GCN(A[4], in_channel, out_channel_list[4], bias=bias, drop=drop, bn=bn, init=init, agg_first=agg_first, attn=attn, part=True)
	def forward(self, x1, x2):
		# import ipdb; ipdb.set_trace()
		output = []
		if 0 in self.branch_list:
			branch_0 = self.branch_0(x2)
			output.append(branch_0)
		if 1 in self.branch_list:
			branch_1 = self.branch_1(x1)
			output.append(branch_1)
		if 2 in self.branch_list:
			branch_2 = self.branch_2(x1)
			output.append(branch_2)
		if 3 in self.branch_list:
			# import ipdb; ipdb.set_trace()
			branch_all = self.branch_all(x1)
			output.append(branch_all)
		if 4 in self.branch_list:
			branch_part = self.branch_part(x1)
			return torch.cat(output, 2), branch_part
		
		return torch.cat(output, 2)

class Block(nn.Module):
	def __init__(self, in_channel, mid_channel, out_channel_list, branch_list, bias=True, drop=None, bn=False, agg_first=True, attn=False):
		super(Block, self).__init__()
		self.linear1 = nn.Linear(in_channel, mid_channel, bias)
		self.linear2 = nn.Linear(in_channel, mid_channel, bias)
		self.pgception = PGception_Layer(mid_channel, out_channel_list, branch_list, bias=bias, drop=drop, bn=bn, agg_first=agg_first, attn=attn)
		self.drop = drop
		self.bn = bn
		if drop:
			self.dropout = nn.Dropout(drop)
		if bn:
			self.batchnorm1 = nn.BatchNorm1d(17)
			self.batchnorm2 = nn.BatchNorm1d(17)

	def forward(self, x1, x2):
		# import ipdb; ipdb.set_trace()
		x1 = self.linear1(x1)
		x2 = self.linear2(x2)
		if self.bn:
			x1 = self.batchnorm1(x1)
			x2 = self.batchnorm2(x2)
		x1 = F.relu(x1)
		x2 = F.relu(x2)
		if self.drop:
			x1 = self.dropout(x1)
			x2 = self.dropout(x2)
		return self.pgception(x1, x2)

class PGception(nn.Module):
	def __init__(self, action_num=24, layers=1, classifier_mod="cat", o_c_l=[64,64,128,128], b_l=[0,1,2,3] ,last_h_c=256, bias=True, drop=None, bn=False, agg_first=True, attn=False):
		super(PGception, self).__init__()
		self.out_channel_list = np.array(o_c_l)
		self.branch_list = b_l
		self.classifier_mod = classifier_mod
		self.drop = drop
		self.bn = bn
		self.layers = layers
		# import ipdb; ipdb.set_trace()
		self.block1 = Block(in_channel= 2, mid_channel=128, out_channel_list=self.out_channel_list, branch_list=self.branch_list, bias=bias, drop=drop, bn=bn, agg_first=agg_first, attn=attn)
		if layers == 2:
			self.block2 = Block(in_channel= sum(self.out_channel_list[self.branch_list]), mid_channel=128, out_channel_list=self.out_channel_list, branch_list=self.branch_list, bias=bias, drop=drop, bn=bn, agg_first=agg_first, attn=attn)
		if classifier_mod == "mean":
			self.classifier = nn.Sequential(
								nn.Linear(sum(self.out_channel_list[self.branch_list]), last_h_c, bias),
								nn.BatchNorm1d(last_h_c),
								nn.ReLU(inplace=True),
								nn.Dropout(drop),
								nn.Linear(last_h_c, action_num, bias),
							)
		if classifier_mod == "cat":
			# add a MLP to reduce the size of channels
			if 4 in b_l:
				# b_l.remove(4) // do not do that
				self.linear = nn.Linear(sum(self.out_channel_list[self.branch_list[:-1]]), 64, bias=True)
			else:
				self.linear = nn.Linear(sum(self.out_channel_list[self.branch_list]), 64, bias=True)
			if drop:
				self.dropout = nn.Dropout(drop)
			if bn:
				self.batchnorm = nn.BatchNorm1d(17)
			self.classifier = nn.Sequential(
								nn.Linear(64*17, last_h_c, bias),
								nn.BatchNorm1d(last_h_c),
								nn.ReLU(inplace=True),
								nn.Dropout(drop),
								nn.Linear(last_h_c, action_num, bias),
							)
			if 4 in b_l:
				self.linear_part = nn.Linear(self.out_channel_list[4], 64, bias=True)
				if bn:
					self.batchnorm_part = nn.BatchNorm1d(5)
				self.classifier_part = nn.Sequential(
									   nn.Linear(64*5, last_h_c, bias),
									   nn.BatchNorm1d(last_h_c),
									   nn.ReLU(inplace=True),
									   nn.Dropout(drop),
									   nn.Linear(last_h_c, action_num, bias),
									)

	def forward(self, x1, x2):
		'''
			x1 is the pose_to_box features,
			x2 is the pose_to_obj_offset features
		'''
		# import ipdb; ipdb.set_trace()
		if 4 in self.branch_list:
			x, x_part = self.block1(x1, x2)
		else:
			x = self.block1(x1, x2)
		if self.layers > 1:
			x = self.block2(x)
		if self.classifier_mod == "mean":
			# import ipdb; ipdb.set_trace()
			x = x.mean(1)
			return self.classifier(x)
		if self.classifier_mod == "cat":
			x = self.linear(x)
			if self.bn:
				x = self.batchnorm(x)
			x = F.relu(x)
			if self.drop:
				x = self.dropout(x)
			# leverage the network for body part, under testing
			if 4 in self.branch_list:
				x_part = self.linear_part(x_part)
				if self.bn:
					x_part = self.batchnorm_part(x_part)
				x_part = F.relu(x_part)
				if self.drop:
					x_part = self.dropout(x_part)
				return self.classifier(x.view(x.shape[0],-1)), self.classifier_part(x_part.view(x_part.shape[0],-1))

			return self.classifier(x.view(x.shape[0],-1))
        
if __name__ == "__main__":
	model = PGception()
	print(model)