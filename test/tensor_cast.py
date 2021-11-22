import torch

x = torch.tensor([float(i) for i in range(100)])
for x in x.detach():
    print(int(x))
