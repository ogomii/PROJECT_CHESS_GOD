import torch
import torch.nn.functional as F

def test_model(model, test_loader, device, loss_fn):
    print(f"Device: {device}")
    print(f"Model:")
    for idx, m in enumerate(model.modules()):
        print(idx, '->', m)

    model.eval()
    with torch.no_grad():
        # check accuracy on test data
        total_loss = 0
        for data in test_loader:
            input, target = data[0].to(device), data[1].to(device)
            logits = model(input)
            loss = loss_fn(logits, target)
            total_loss += loss.item()
        print(f'Avg loss of the network on test dataset: {total_loss / len(test_loader)}')