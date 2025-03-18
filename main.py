import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data import SimpleSeqDataset
from model import ThreeLayerNet

STUDENT_ID = "20242056"
N_CLASS = 19
MAX_LEN = 20

if torch.cuda.is_available():
	torch.set_default_tensor_type(torch.cuda.FloatTensor)

def train(train_loader, model, loss_fn, optimizer, scheduler, n_epoch=100, tol=1e-6):
	print(f'------ Train (n_epoch={n_epoch}, tol={tol}) ------')
	print('%6s%14s%14s' % ('epoch', 'loss_epoch', 'delta'))

	loss_epoch_old, delta = 100, 100
	for epoch in range(n_epoch):
		loss_epoch = 0
		for x, y in train_loader:
			optimizer.zero_grad()
			y_pred = model(x)
			loss = loss_fn(y_pred, y)
			loss.backward()
			loss_epoch += loss.item()
			optimizer.step()
		scheduler.step()

		loss_epoch /= len(train_loader)
		delta = abs(loss_epoch - loss_epoch_old)
		print(f'{epoch:6d}{loss_epoch:14f}{delta:14f}')
		if delta < tol: break
		else: loss_epoch_old = loss_epoch

	print(f'--------------------------------------------', end='\n\n')

def test(test_loader, model, label_dict):
	label_dict_inv = {v:k for k, v in label_dict.items()}
	model.eval()

	df = []
	with torch.no_grad():
		for i, (x, _) in enumerate(test_loader):
			y_pred = model(x)
			df.append([f'S{i:03d}', label_dict_inv[y_pred.argmax(dim=1).item()]])
	df = pd.DataFrame(df, columns=['id', 'pred'])
	
	return df

def main():
	print('Load Data...', end=' ')
	train_dset = SimpleSeqDataset('./dataset/simple_seq.train.csv', max_len=MAX_LEN)
	test_dset  = SimpleSeqDataset('./dataset/simple_seq.test.csv',  max_len=MAX_LEN, token_dict=train_dset.token_dict)
	print('Done', end='\n\n')

	train_loader = DataLoader(train_dset, batch_size=16, shuffle=True,  drop_last=False)
	test_loader  = DataLoader(test_dset,  batch_size=1,  shuffle=False, drop_last=False)

	model = ThreeLayerNet(train_dset.data.shape[1], 1000, 100, N_CLASS)
	loss_fn = nn.CrossEntropyLoss()

	optimizer = optim.SGD(model.parameters(), lr=1e-1)
	scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

	train(train_loader, model, loss_fn, optimizer, scheduler, n_epoch=100, tol=1e-3)
	df1 = test(test_loader, model, train_dset.label_dict)
	df1.to_csv(f'{STUDENT_ID}_simple_seq.p1.answer.csv', index=False)

if __name__ == "__main__":
	main()
