import torch
import torchvision
import torchvision.transforms as transforms

def load_data(data_path, batch_size):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_set = torchvision.datasets.MNIST(root=data_path, train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    val_set = torchvision.datasets.MNIST(root=data_path, train=False, download=True, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)

def log_metrics(epoch, loss, log_path):
    with open(log_path, 'a') as log_file:
        log_file.write(f'Epoch: {epoch}, Loss: {loss}\n')
