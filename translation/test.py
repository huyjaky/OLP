import torch
import pandas as pd 

A1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
B = torch.tensor([[1, 1, 0], [0, 0, 1]])
print(A1 == B)

df = pd.DataFrame()



