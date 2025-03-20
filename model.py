import numpy as np

import torch
import torch.nn as nn

class Linear(nn.Module):
	def __init__(self, input_size, output_size):
		super().__init__()	

		self.w = nn.Parameter(torch.randn(output_size, input_size))
		self.b = nn.Parameter(torch.zeros(output_size))
		nn.init.xavier_uniform_(self.w)

	def forward(self, x):
		return x @ self.w.T + self.b

class Embedding(nn.Module):
	def __init__(self, token_size, emb_dim, pad_idx=0):
		super().__init__()

		self.w = nn.Parameter(torch.randn(token_size, emb_dim))
		nn.init.xavier_uniform_(self.w)

		# Padding
		self.pad_idx = pad_idx
		with torch.no_grad(): self.w[pad_idx] = 0
		self.w.register_hook(self.padding)

	def padding(self, grad):
		grad[self.pad_idx] = 0
		return grad

	def forward(self, x):
		return self.w[x]

class ThreeLayerNetOneHot(nn.Module):
	def __init__(self, max_len, token_size, hidden1_size, hidden2_size, output_size):
		super().__init__()	

		self.l1 = Linear(max_len * token_size, hidden1_size)
		self.l2 = Linear(hidden1_size, hidden2_size)
		self.l3 = Linear(hidden2_size, output_size)

		self.bn1 = nn.BatchNorm1d(hidden1_size)
		self.bn2 = nn.BatchNorm1d(hidden2_size)

	def forward(self, x):
		x = self.l1(x)
		x = self.bn1(x)
		x = torch.sigmoid(x)

		x = self.l2(x)
		x = self.bn2(x)
		x = torch.sigmoid(x)

		x = self.l3(x)
		return x

class ThreeLayerNetWordEmb(nn.Module):
	def __init__(self, max_len, emb_dim, token_size, hidden1_size, hidden2_size, output_size):
		super().__init__()	

		self.emb = Embedding(token_size, emb_dim)
			
		self.l1 = Linear(max_len * emb_dim, hidden1_size)
		self.l2 = Linear(hidden1_size, hidden2_size)
		self.l3 = Linear(hidden2_size, output_size)

		self.bn1 = nn.BatchNorm1d(hidden1_size)
		self.bn2 = nn.BatchNorm1d(hidden2_size)

	def forward(self, x):
		x = self.emb(x).flatten(start_dim=1)

		x = self.l1(x)
		x = self.bn1(x)
		x = torch.sigmoid(x)

		x = self.l2(x)
		x = self.bn2(x)
		x = torch.sigmoid(x)

		x = self.l3(x)
		return x
