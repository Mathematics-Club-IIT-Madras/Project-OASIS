import numpy as np
import torch
import csv


b_2 = 0
b_1 = 3
b_0 = 0.5

x_1 = torch.rand(100).view(-1,1) 
x_2 = torch.rand(100).view(-1,1) 

e = torch.normal(0,0.4,(100,)).view(-1,1)

y = b_0 + b_1*x_1 + b_2*x_2 + e


data = torch.cat((x_1, x_2, y), dim=1).numpy()
print(data[:5])


with open('/Users/kailashanand/Documents/Developer/MC_OASIS/Task_Week4/amazing_data_0.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(data)
        


