import argparse
import numpy as np
import pandas as pd
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler

from data import SimpleSeqDataset
from model import ThreeLayerNetOneHot, ThreeLayerNetWordEmb

parser = argparse.ArgumentParser()
parser.add_argument('--onehot', action='store_true')
args = parser.parse_args()

STUDENT_ID = "20242056"
N_CLASS = 19
MAX_LEN = 20
EMB_DIM = 0 if args.onehot else 10

if torch.cuda.is_available():
	torch.set_default_tensor_type(torch.cuda.FloatTensor)

def train(train_loader, model, loss_fn, optimizer, scheduler, n_epoch=100, tol=1e-6, pat=3):
	print(f'---------- Train (n_epoch={n_epoch}, tol={tol}) ----------')
	print('%6s%14s%14s%14s' % ('epoch', 'acc', 'loss', 'delta'))

	loss_old, delta, cnt = 100, 100, 0
	for epoch in range(n_epoch):
		acc, loss = 0, 0
		for x, y in train_loader:
			optimizer.zero_grad()
			y_pred = model(x)

			acc_batch = (y_pred.argmax(dim=1) == y).float().mean()
			acc += acc_batch.item()

			loss_batch = loss_fn(y_pred, y)
			loss_batch.backward()
			loss += loss_batch.item()

			optimizer.step()
		scheduler.step()

		# Early stopping
		acc /= len(train_loader)
		loss /= len(train_loader)
		delta = abs(loss - loss_old)
		print(f'{epoch:6d}{acc:14f}{loss:14f}{delta:14f}')
		if delta < tol:
			if cnt < pat: break
			else: cnt += 1
		else: loss_old = loss

	print(f'----------------------------------------------------', end='\n\n')

def test(test_loader, model, label_dict):
	label_dict_inv = {v:k for k, v in label_dict.items()}

	df = []
	model.eval()
	with torch.no_grad():
		for i, (x, _) in enumerate(test_loader, start=1):
			y_pred = model(x)
			df.append([f'S{i:03d}', label_dict_inv[y_pred.argmax(dim=1).item()]])
	df = pd.DataFrame(df, columns=['id', 'pred'])
	
	return df

def main():
	print('Load Data...', end=' ')
	train_dset = SimpleSeqDataset('./dataset/simple_seq.train.csv', max_len=MAX_LEN, onehot=args.onehot)
	token_dict, label_dict = train_dset.token_dict, train_dset.label_dict
	test_dset = SimpleSeqDataset('./dataset/simple_seq.test.csv', max_len=MAX_LEN, onehot=args.onehot, token_dict=token_dict)
	print('Done', end='\n\n')

	# Class weight to deal with imbalance
	class_cnt = dict(sorted(Counter(train_dset.label.to('cpu').numpy()).items()))
	class_weight = {k: 1. / v for k, v in class_cnt.items()}
	sample_weight = torch.tensor([class_weight[label] for label in train_dset.label.to('cpu').numpy()])
	sampler = WeightedRandomSampler(sample_weight, len(sample_weight))
	#print(class_cnt, class_weight, sep='\n', end='\n\n')

	train_loader = DataLoader(train_dset, batch_size=8, drop_last=True, sampler=sampler)
	test_loader = DataLoader(test_dset, batch_size=1, drop_last=False, shuffle=False)

	if args.onehot:
		model = ThreeLayerNetOneHot(MAX_LEN, len(token_dict), 1000, 100, N_CLASS)
	else:
		model = ThreeLayerNetWordEmb(MAX_LEN, EMB_DIM, len(token_dict), 1000, 100, N_CLASS)

	loss_fn = nn.CrossEntropyLoss()
	#optimizer = optim.SGD(model.parameters(), lr=1e-1)
	optimizer = optim.Adam(model.parameters(), lr=1e-2)
	scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
	
	train(train_loader, model, loss_fn, optimizer, scheduler, n_epoch=100, tol=1e-5, pat=3)

	df = test(test_loader, model, label_dict)
	df.to_csv(f'{STUDENT_ID}_simple_seq.p{2-args.onehot}.answer.csv', index=False)

if __name__ == "__main__":
	main()
