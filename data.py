import numpy as np

import torch
from torch.utils.data import Dataset

class SimpleSeqDataset(Dataset):
	def __init__(self, fn, max_len=20, token_dict=None):
		self.token_dict = token_dict

		# Load sequence
		data = [list(filter(None, line.strip().split(','))) for line in open(fn, 'r')]

		if self.token_dict: # Replace all unknown tokens to 'UNK'
			token_set = set(token_dict)
			data = [[word if word in token_set else 'UNK' for word in sample] for sample in data]
			label = np.zeros(len(data))
		else: # Gen dictionary
			data, label = zip(*[(sample[:-1], sample[-1]) for sample in data])
			self.token_dict = {token:i for i, token in enumerate(list(np.unique(sum(data, []))) + ['UNK'])}
			self.label_dict = {label:i for i, label in enumerate(np.unique(label))}
			label = [self.label_dict[l] for l in label]

		# Transform sequence to one-hot representation and add padding
		I = np.eye(len(self.token_dict))
		data = [np.pad(I[[self.token_dict[word] for word in sample[:max_len]]],\
				pad_width=((0, max(0, max_len - len(sample))), (0, 0)))\
				for sample in data]

		# Vectorize
		self.data  = torch.tensor(data,  dtype=torch.float, requires_grad=True).flatten(start_dim=1)
		self.label = torch.tensor(label, dtype=torch.long)

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		return self.data[idx], self.label[idx]
