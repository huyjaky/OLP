import pandas
import torch

A = torch.tensor([[1, 2, 3], [4, 5, 6]])
A[:, :2] = A[:, :2] * 10
print(A)
