import matplotlib.pyplot as plt
import torch


class SobrietyTrainer:
    def __init__(self, model, train_loader, val_loader, test_loader, optimizer, criterion, scheduler, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.model.to(device)
        self.scheduler = scheduler
        self.val_accuracy = []
        self.test_accuracy = None

    def train(self, num_epochs: int, epochs_per_save: int = 1):
        for epoch in range(num_epochs):
            self.train_epoch()
            self.test_epoch(validate=True)
            print(f"Epoch {epoch + 1}: Validation accuracy = {self.val_accuracy[-1]: .3f}")
            if epoch % epochs_per_save == 0:
                self.save_model(f"model_epoch_{epoch + 1}.pt")
        self.graph_accuracy_by_epoch()
        self.test_epoch(validate=False)
        print(f"Test accuracy = {self.test_accuracy: .3f}")

    def train_epoch(self):
        self.model.train()
        for inputs, labels in self.train_loader:
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
        self.scheduler.step()

    def test_epoch(self, validate: bool = True):
        self.model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            loader = self.val_loader if validate else self.test_loader
            for inputs, labels in loader:
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            acc = correct / total
            if validate:
                self.val_accuracy.append(acc)
            else:
                self.test_accuracy = acc

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
