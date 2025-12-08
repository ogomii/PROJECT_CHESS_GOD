import torch
import torch.nn.functional as F

def train_loop(model, n_epochs, trainloader, testloader, optim, device, writer, loss_fn):
    with torch.no_grad():
        model.eval()
        test_loss = []
        for i, data in enumerate(trainloader, 0):
            inputs, target = data[0].to(device), data[1].to(device)
            B, C, H, W = inputs.shape
            logits = model(inputs)
            loss = loss_fn(logits, target)
            test_loss.append(loss.item())
        train_loss_avg = sum(test_loss) / len(test_loss)  
        print(f'Initialized network train_loss: {train_loss_avg}')
        model.train()

    print(f'------Commencing training:------')
    epoch_loss_avg_arr = []
    model.train()
    for epoch in range(n_epochs):  # loop over the dataset multiple times
        epoch_loss = []
        print(f"Running epoch: {epoch}")

        # train on training data
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            
            logits = model(inputs)

            loss = loss_fn(logits, target)
            epoch_loss.append(loss.item())
            optim.zero_grad()
            loss.backward()
            optim.step()
        epoch_loss_avg = sum(epoch_loss) / len(epoch_loss) 
        epoch_loss_avg_arr.append(epoch_loss_avg)
        writer.add_scalar('Loss/train', epoch_loss_avg, epoch)

        # check performance on eval data
        if (epoch % 4 == 0) or (epoch == n_epochs-1):
            model.eval()
            test_loss = []
            for i, data in enumerate(testloader, 0):
                inputs, target = data[0].to(device), data[1].to(device)
                B, C, H, W = inputs.shape
                logits = model(inputs)
                test_loss.append(loss_fn(logits, target).item())
            test_loss_avg = sum(test_loss) / len(test_loss)          
            writer.add_scalar('Loss/test', test_loss_avg, epoch)
            print(f'epoch: {epoch} train_loss: {epoch_loss_avg}, val_loss: {test_loss_avg}')
            model.train()
        
        # early stopping
        if epoch > 5:
            if min(epoch_loss_avg_arr[-5:-3]) < (epoch_loss_avg + 0.01):
                print(f"Early stopping on epoch {epoch} due to current loss_avg: {epoch_loss_avg} compared to last 4: {epoch_loss_avg_arr[-5:-1]}")
                break
        
    writer.close()

    def check_accuracy(model, dataloader, acc_type='test'):
        model.eval()
        with torch.no_grad():
            # check accuracy on data
            correct = 0
            total = 0
            for data in dataloader:
                images, target = data[0].to(device), data[1].to(device)
                B, C, H, W = images.shape
                logits = model(images)
                _, predicted = torch.max(logits, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
            print(f'Accuracy of the network on {acc_type} images: {100 * correct // total} %')
        model.train()

    check_accuracy(model, trainloader, 'train')
    check_accuracy(model, testloader, 'test')