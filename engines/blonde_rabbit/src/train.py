import torch
from torch.utils.tensorboard import SummaryWriter  # loss tracking
from torch.utils.data import DataLoader
from model import BlondeRabbit, Config
from tools.train_loop import train_loop
from tools.data import ChessDataset
from tools.common import TrainingConfig
import datetime

device = 'cuda' if torch.cuda.is_available() else 'cpu'

training_config = TrainingConfig(
                    n_epochs=20, 
                    batch_size=3200, 
                    learning_rate=0.001,
                    loss_fn=torch.nn.MSELoss(),
                    early_stop=True,
                    model_save_path=f'engines/blonde_rabbit/weights/blonde_rabbit_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.pth'
                    )

print("Loading testset...")
test_dataset = ChessDataset('data/nnue/test')
print("Loading trainset...")
train_dataset = ChessDataset('data/nnue/train')
# train_dataset = ChessDataset('data/nnue/test') # temporary for quick testing

print("Creating batched dataloaders...")
trainloader = DataLoader(train_dataset, batch_size=training_config.batch_size, shuffle=True, num_workers=2)
testloader = DataLoader(test_dataset, batch_size=training_config.batch_size, shuffle=False, num_workers=2)

print("Creating model...")
m = BlondeRabbit(Config)
model = m.to(device)
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

optim = torch.optim.Adam(model.parameters(), lr=training_config.learning_rate)
writer = SummaryWriter()

print("Starting training loop...")
train_loop(model, training_config, trainloader, testloader, optim, device, writer)