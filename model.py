import torch.nn as nn

""" Define your own model """
import torch
import torchvision
class FewShotModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. Load pretrained model
        self.model = torchvision.models.resnet101(pretrained=True, progress=True)
        # 2. Fix parameters
        for param in self.model.parameters():
            param.requires_grad = False
        # 3. Initialize a last layer for fine-tunning
        self.model.fc = nn.Linear(self.model.fc.in_features, 1000) 

        '''
        # Bad Result for multi-last layer
        self.model.fc = nn.Sequential(
              nn.Linear(model.fc.in_features, 1000),
              nn.ReLU(), nn.Dropout(0.5),
              nn.Linear(1000, 512),
              nn.ReLU(), nn.Dropout(0.5),
              nn.Linear(512, 512),
           ) 
        '''
    def forward(self, x):
        return self.model(x)


class FewShotModel_ensemble(nn.Module):
    def __init__(self):
        super().__init__()
        # Load pretrained model
        self.model1 = torchvision.models.resnet101(pretrained=True, progress=True)
        self.model2 = torchvision.models.resnext101_32x8d(pretrained=True, progress=True)
        self.model3 = torchvision.models.densenet161(pretrained=True, progress=True)
        # Fix parameters
        for param in self.model1.parameters():
            param.requires_grad = False
        for param in self.model2.parameters():
            param.requires_grad = False
        for param in self.model3.parameters():
            param.requires_grad = False
        # Initialize a last layer for fine-tuning
        self.model1.fc = nn.Linear(self.model1.fc.in_features, 1000)
        self.model2.fc = nn.Linear(self.model2.fc.in_features, 1000)
        self.model3.classifier = nn.Linear(self.model3.classifier.in_features, 1000)

    def forward(self, x):
        # Stack each models output features and take average it.
        y = torch.stack([self.model1(x),self.model2(x),self.model3(x)], dim=1)
        return torch.mean(y, dim=1)
