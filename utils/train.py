import torch
import copy
import numpy as np

def test_boolean(network, loss, dataloader, dev):
    network.eval()
    total = 0
    correct1 = 0
    total_loss = 0
    with torch.no_grad():
        for idx, (data, target) in enumerate(dataloader):
            data = data.to(dev)
            target = target.to(dev)
            output = network(data.float())
            output = output[-1]
            total_loss += loss(output,target,dev)
            inds = output > 0.5
            output = inds.float()
            total += torch.numel(target)
            correct1 += torch.sum(output==target)
    acc = 100.0 * correct1 / total
    total_loss = total_loss / idx

    print('Top 1 Accuracy =', acc)
    print('Average Loss =', total_loss)

    return total_loss.detach().cpu().numpy(), acc.detach().cpu().numpy()

def test_mnist(network, loss, dataloader, dev):
    network.eval()
    total = 0
    correct1 = 0
    correct5 = 0
    with torch.no_grad():
        for idx, (data, target) in enumerate(dataloader):
            data = data.to(dev)
            target = target.to(dev)
            output = network(data)
            total += loss(output[-1], target, dev).item() * data.size(0)
            _, pred = output[-1].topk(5, dim=1)
            correct = pred.eq(target.view(-1,1).expand_as(pred))
            correct1 += correct[:,:1].sum().item()
            correct5 += correct[:,:5].sum().item()
    avg_loss = total / len(dataloader.dataset)
    acc1 = 100.0 * correct1 / len(dataloader.dataset)
    acc5 = 100.0 * correct5 / len(dataloader.dataset)

    print('Top 1 Accuracy =', acc1)
    print('Top 5 Accuracy =', acc5)
    print('Average Loss =', avg_loss)

    return avg_loss, acc1

def train_network(network, loss, optimizer, train_loader, validation_loader, test_loader, dev, epochs, scheduler, task='boolean'):
    
    train_curve = []
    accuracy1 = []
    test_loss = []
    avg_loss_min = 1000.0
    acc_max = 0.0

    for epoch in range(epochs):
        network.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(dev)
            target = target.to(dev)
            optimizer.zero_grad()
            output = network(data.float())
            batch_loss = loss(output[-1], target, dev)
            train_loss += batch_loss.item() * data.size(0)
            batch_loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), batch_loss.item()))

        train_curve.append(train_loss / len(train_loader.dataset))
        if task=='mnist_10way':
            avg_loss, acc1 = test_mnist(network, loss, validation_loader, dev)
        else:
            avg_loss, acc1 = test_boolean(network, loss, validation_loader, dev)
        if avg_loss < 1000.0:
            if avg_loss_min>=avg_loss and acc1 >= acc_max:
                net = copy.deepcopy(network)
                avg_loss_min = avg_loss
                acc_max = acc1
            elif acc1>acc_max: ## change to 
                net = copy.deepcopy(network)
                avg_loss_min = avg_loss
                acc_max = acc1
        else:
            if acc1 >= acc_max:
                net = copy.deepcopy(network)
                acc_max = acc1
        accuracy1.append(acc1)
        test_loss.append(avg_loss)

        scheduler.step()

    avg_loss, acc1 = test_boolean(net, loss, test_loader, dev)

    return net, train_curve, test_loss, accuracy1, avg_loss, acc1, epochs
