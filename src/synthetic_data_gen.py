import numpy as np
import torch
import csv

torch.manual_seed(0)

N = 1e5
P = 5

noise_mean = 0.0
noise_std = 0.05

attack_data = True
attack_data_mean = 1000.0
attack_data_std = 1000
num_adversaries = int(N/2)

data_mean = torch.tensor(0)
data_std = torch.tensor(1)
        

# b = torch.rand(size=(1,P+1)).view(-1, 1) # goes from b_0, b_1, b_2...b_P
b = torch.tensor([1, 0.5, 2, 3, 4, 5]).view(-1, 1)


if (b.shape[0] != P+1):
    print("Size mis-match")
    quit()

X = torch.normal(data_mean, data_std, size=(N,P+1))
X[:,0] = torch.ones(N)

e = torch.normal(noise_mean, noise_std, (N,)).view(-1, 1)
y = X @ b + e


if (attack_data):
    adversaries = torch.normal(attack_data_mean, attack_data_std, (num_adversaries,1))
    attack_idxs = torch.randperm(N)[:num_adversaries]

    y[attack_idxs] += adversaries
    


data = torch.cat((X[:,1:], y), dim=1).numpy()

with open('/Users/kailashanand/Documents/Developer/MC_OASIS/TORRENT/amazing_data_0.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(data)