from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch

# transform 없이 tensor만 받기
transform = transforms.ToTensor()
trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
loader = DataLoader(trainset, batch_size=50000, shuffle=False)

data = next(iter(loader))[0]  # (50000, 3, 32, 32)
mean = data.mean(dim=(0, 2, 3))
std = data.std(dim=(0, 2, 3))

print("CIFAR-100 평균:", mean)
print("CIFAR-100 표준편차:", std)

# CIFAR-100 평균: tensor([0.5071, 0.4866, 0.4409])
# CIFAR-100 표준편차: tensor([0.2673, 0.2564, 0.2762])