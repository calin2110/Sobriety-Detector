import torch.nn as nn
import torchvision


class SobrietyModel(nn.Module):
    def __init__(self, num_classes: int):
        super(SobrietyModel, self).__init__()
        # load the pretrained model
        self.model = torchvision.models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V2', progress=True)

        # freeze the parameters of the pretrained model
        for param in self.model.parameters():
            param.requires_grad = False

        num_ftrs = self.model.fc.in_features
        # replace the final fully connected layer with three new fully connected layers
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.model(x)
