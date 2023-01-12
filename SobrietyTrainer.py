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

    def train(self, num_epochs: int, epochs_per_save: int = 1):
        for epoch in range(num_epochs):
            self.train_epoch()
            self.test_epoch()
            print(f"Epoch {epoch + 1}: Validation accuracy = {self.val_accuracy[-1]: .3f}")
            if epoch % epochs_per_save == 0:
                self.save_model(f"model_epoch_{epoch + 1}.pt")
        self.graph_accuracy_by_epoch()
        self.test_model()

    def train_epoch(self):
        self.model.train()
        nr_batches = len(self.train_loader)
        batch = 0
        for inputs, labels in self.train_loader:
            batch += 1
            print(f"Batch {batch}/{nr_batches}", end='\r')
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
        self.scheduler.step()

    def test_epoch(self):
        self.model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for inputs, labels in self.val_loader:
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            acc = correct / total
            self.val_accuracy.append(acc)

    def test_model(self):
        self.model.eval()
        nr_batches = len(self.train_loader)
        batch = 0
        with torch.no_grad():
            FP = 0
            FN = 0
            TP = 0
            TN = 0
            for inputs, labels in self.test_loader:
                batch += 1
                print(f"Test batch {batch}/{nr_batches}", end='\r')
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)

                FP += ((predicted == 1) & (labels == 0)).sum().item()
                FN += ((predicted == 0) & (labels == 1)).sum().item()
                TP += ((predicted == 1) & (labels == 1)).sum().item()
                TN += ((predicted == 0) & (labels == 0)).sum().item()
            print(f"Test set: FP: {FP}, FN: {FN}, TP: {TP}, TN: {TN}")
            acc = (TP + TN) / (TP + TN + FP + FN)
            recall = TP / (TP + FN)
            precision = TP / (TP + FP)
            f1_score = 2 * (precision * recall) / (precision + recall)
            print(f"Accuracy: {acc}, Recall: {recall}, Precision: {precision}, F1 Score: {f1_score}")

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
