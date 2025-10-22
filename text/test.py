from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import torch
train_set = MNIST(root="/tmp/data/MNIST", train=True, transform=ToTensor(), download=True)
val_set = MNIST(root="/tmp/data/MNIST", train=False, transform=ToTensor(), download=False)

train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=64, shuffle=True, pin_memory=torch.cuda.is_available(), num_workers=4
)
val_loader = torch.utils.data.DataLoader(
    val_set, batch_size=64, shuffle=False, pin_memory=torch.cuda.is_available(), num_workers=4
)

for sample in train_loader:
    print(len(sample))
    break


