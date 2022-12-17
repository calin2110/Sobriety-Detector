import matplotlib.pyplot as plt
import torch


class SobrietyTrainer:
    def __init__(self, model, train_loader, val_loader, optimizer, criterion, scheduler, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.model.to(device)
        self.scheduler = scheduler
        self.val_accuracy = []

    def train(self, num_epochs: int, epochs_per_save: int = 1):
        for epoch in range(num_epochs):
            self.train_epoch(epoch)
            self.test_epoch(epoch)
            if epoch % epochs_per_save == 0:
                self.save_model(f"model_epoch_{epoch + 1}.pt")
        self.graph_accuracy_by_epoch()

    def train_epoch(self, epoch: int):
        self.model.train()
        for inputs, labels in self.train_loader:
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
        self.scheduler.step()

    def test_epoch(self, epoch: int):
        self.model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for inputs, labels in self.val_loader:
                inputs, label = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            val_acc = correct / total
            self.val_accuracy.append(val_acc)
            print(f"Epoch {epoch + 1}: Validation accuracy = {val_acc: .3f}")

    def save_model(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path: str):
        self.model.load_state_dict(torch.load(path))

    def graph_accuracy_by_epoch(self):
        plt.plot(self.val_accuracy)
        plt.xlabel("Epoch")
        plt.ylabel("Validation Accuracy")
        plt.title("Validation Accuracy over Time")
        plt.show()
