import torch
import torch.nn as nn
from torchvision import models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DenseNet121(nn.Module):
    def __init__(self, count_classes, is_trained=True):
        super(DenseNet121, self).__init__()

        self.model = models.densenet121(pretrained=is_trained)

        kernel_count = self.model.classifier.in_features

        self.model.classifier = nn.Sequential(nn.Linear(kernel_count, count_classes), nn.Softmax())

    def forward(self, x):
        x = self.model(x)

        return x
