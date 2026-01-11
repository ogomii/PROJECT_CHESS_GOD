import torch
from torch.utils.tensorboard import SummaryWriter  # loss tracking
from torch.utils.data import DataLoader
from model import BlondeRabbit, Config
from tools.train_loop import train_loop
from tools.data import ChessDataset
from tools.common import TrainingConfig
import datetime
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

training_config = TrainingConfig(
                    n_epochs=20, 
                    batch_size=1600, 
                    learning_rate=1e-5,
                    loss_fn=torch.nn.MSELoss(),
                    patience=5,
                    early_stop=True,
                    weight_decay=1e-4,
                    model_save_path=f'engines/blonde_rabbit/weights/blonde_rabbit_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.pth'
                    )

print("Loading testset...")
test_dataset = ChessDataset('data/nnue/', 'test')
print("Loading trainset...")
# train_dataset = ChessDataset('data/nnue/test') # temporary for quick testing
train_dataset = ChessDataset('data/nnue/', 'train')
# normalize dataset to stabilize training, denormalize predictions on inference
train_targets = train_dataset.targets  # np array
target_mean = np.mean(train_targets)
target_std = np.std(train_targets)
print(f"Target mean: {target_mean}, std: {target_std}")
train_dataset.normalize(target_mean, target_std)
test_dataset.normalize(target_mean, target_std)

print("Creating batched dataloaders...")
trainloader = DataLoader(train_dataset, batch_size=training_config.batch_size, shuffle=True, num_workers=2)
testloader = DataLoader(test_dataset, batch_size=training_config.batch_size, shuffle=False, num_workers=2)

print("Creating model...")
m = BlondeRabbit(Config, target_mean=train_dataset.target_mean, target_std=train_dataset.target_std)
model = m.to(device)
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

optim = torch.optim.AdamW(
                        model.parameters(), 
                        lr=training_config.learning_rate, 
                        weight_decay=training_config.weight_decay
                        )
writer = SummaryWriter()

print("Starting training loop...")
train_loop(model, training_config, trainloader, testloader, optim, device, writer)