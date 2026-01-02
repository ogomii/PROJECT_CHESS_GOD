import torch
from torch.utils.tensorboard import SummaryWriter  # loss tracking
from torch.utils.data import DataLoader
from model import BlondeRabbit, Config
from tools.train_loop import train_loop
from tools.data import ChessDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

n_epochs = 5
batch_size = 160 # multiples of 16 only due to dataset size
learning_rate = 0.01

# TODO: preprocess data into tensors and save as binary files for faster loading
print("Creating testset...")
test_dataset = ChessDataset('data/nnue/test.csv', Config)
print("Creating trainset...")
train_dataset = ChessDataset('data/nnue/train.csv', Config)

print("Creating dataloaders...")
trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

print("Creating model...")
m = BlondeRabbit(Config)
model = m.to(device)
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
writer = SummaryWriter()

print("Starting training loop...")
train_loop(model, trainloader, testloader, optim, n_epochs, device, writer, loss_fn=torch.nn.MSELoss())