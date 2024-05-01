import torch

def accuracy(loader, model, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    model.train()
    return correct / total

def accuracyCustom(loader, memloader, model, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, mem in zip(loader, memloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            mem, _ = mem
            mem = mem.to(device)
            outputs = model(inputs, mem)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    model.train()
    return correct / total

evalDic = {
    'accuracy': accuracy,
    'accuracyCustom': accuracyCustom
}