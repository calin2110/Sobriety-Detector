import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from SobrietyDataset import SobrietyDataset
from SobrietyModel import SobrietyModel
from SobrietyTrainer import SobrietyTrainer


def main():
    # define the number of epochs
    num_epochs = 10

    # define the batch size
    batch_size = 64

    # define the train dataset
    train_dataset = SobrietyDataset(root_dir="data/train", csv_file="data/train.csv", expected_size=(224, 224))

    # define the train loader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # define the validation dataset
    validation_dataset = SobrietyDataset(root_dir="data/validation", csv_file="data/validation.csv",
                                         expected_size=(224, 224))

    # define the validation loader
    val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)

    # define the test dataset
    test_dataset = SobrietyDataset(root_dir="data/test", csv_file="data/test.csv", expected_size=(224, 224))

    # define the test loader
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    sobriety_model = SobrietyModel(
        num_classes=2
    )

    # move the model to GPU if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(sobriety_model.parameters(), lr=0.001, weight_decay=1e-5)

    # define the learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    sobriety_trainer = SobrietyTrainer(
        model=sobriety_model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler
    )

    sobriety_trainer.train(num_epochs=num_epochs)


if __name__ == "__main__":
    main()
