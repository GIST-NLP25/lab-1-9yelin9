import torch
import numpy as np
import pandas as pd

STUDENT_ID = "20242056"
N_CLASS = 19
MAX_LEN = 20

def load_data(fn, token_dict=None):
	# Load sequence
	data, target = [list(filter(None, line.strip().split(','))) for line in open(fn, 'r')], []

	if token_dict: # Replace all unknown tokens to 'UNK'
		token_set = set(token_dict)
		data = [[word if word in token_set else 'UNK' for word in sample] for sample in data]
	else: # Gen dictionary
		data, target = zip(*[(sample[:-1], sample[-1]) for sample in data])
		token_list = list(np.unique(sum(data, []))) + ['UNK']
		token_dict = {token:i for i, token in enumerate(token_list)}

	# Transform sequence to one-hot representation and add padding
	I = np.eye(len(token_dict))
	data = [np.pad(I[[token_dict[word] for word in sample[:MAX_LEN]]],\
			pad_width=((0, max(0, MAX_LEN - len(sample))), (0, 0)))\
			for sample in data]

	# Vectorize
	data = torch.tensor(data, requires_grad=True).flatten(start_dim=1)

	return data, target, token_dict

def save_data(df1, df2):
	# EXAMPLE
	# Save the data to a csv file
	# You can change function
	# BUT you should keep the file name as "{STUDENT_ID}_simple_seq.p#.answer.csv"
	df1.to_csv(f'{STUDENT_ID}_simple_seq.p1.answer.csv')
	df2.to_csv(f'{STUDENT_ID}_simple_seq.p2.answer.csv')

def main():
	#save_data(df1, df2)
	train_data, train_target, token_dict = load_data('./dataset/simple_seq.train.csv')
	test_data, _, _ = load_data('./dataset/simple_seq.test.csv', token_dict=token_dict)

	

if __name__ == "__main__":
	main()
