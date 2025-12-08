from torch.utils.tensorboard import SummaryWriter  # loss tracking
import torchvision
import torchvision.transforms as transforms
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

n_epochs = 5
batch_size = 80 # multiples of 16 only due to dataset size
learning_rate = 0.0005

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='/home/ogomi/AI/ViT/', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='/home/ogomi/AI/ViT/', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

model = m.to(device)
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
writer = SummaryWriter()