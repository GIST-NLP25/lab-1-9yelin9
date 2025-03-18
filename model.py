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

class ThreeLayerNet(nn.Module):
	def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
		super().__init__()	

		self.l1 = Linear(input_size,   hidden1_size)
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
