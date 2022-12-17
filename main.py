import torch
import torchvision
import torch.nn as nn


from SobrietyModel import SobrietyModel
from SobrietyTrainer import SobrietyTrainer


def main():
    # define the number of epochs
    num_epochs = 10

    # define the batch size
    batch_size = 64

    # define the train loader
    # TODO: create the train loaded
    train_loader = None
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # define the validation loader
    # TODO: create the validation loader
    val_loader = None
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # TODO: discuss the number of resulting classes
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
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler
    )

    sobriety_trainer.train(num_epochs=num_epochs)


if __name__ == "__main__":
    main()
