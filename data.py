import numpy as np

import torch
from torch.utils.data import Dataset

class SimpleSeqDataset(Dataset):
	def __init__(self, fn, max_len=20, token_dict=None, onehot=True):
		self.token_dict = token_dict

		# Load sequence
		data = [list(filter(None, line.strip().split(','))) for line in open(fn, 'r')]

		if self.token_dict: # Replace all unknown tokens to 'UNK'
			token_set = set(token_dict)
			data = [[word if word in token_set else 'UNK' for word in sample] for sample in data]
			label = np.zeros(len(data))
		else: # Gen dictionary
			data, label = zip(*[(sample[:-1], sample[-1]) for sample in data])
			self.token_dict = {token:i for i, token in enumerate(['PAD', 'UNK'] + list(np.unique(sum(data, []))))}
			self.label_dict = {label:i for i, label in enumerate(np.unique(label))}
			label = [self.label_dict[l] for l in label]
		self.label = torch.tensor(label, dtype=torch.long)

		# Transform sequence to token dictionary indices and add padding
		data = [[self.token_dict[word] for word in sample[:max_len]] + [0] * max(0, max_len - len(sample)) for sample in data]

		if onehot:
			I = np.eye(len(self.token_dict)); I[0] = 0 # Padding
			data = [I[sample] for sample in data]
			self.data = torch.tensor(data, dtype=torch.float, requires_grad=True).flatten(start_dim=1)
		else:
			self.data = torch.tensor(data, dtype=torch.long, requires_grad=False)

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		return self.data[idx], self.label[idx]
