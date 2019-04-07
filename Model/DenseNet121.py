import torchvision
import torch.nn as nn

class DenseNet121(nn.Module):

    def __init__(self,return_size):

        super(DenseNet121,self).__init__()
        self.denseNet121 = torchvision.models.densenet121(pretrained=True)
        features = self.denseNet121.classifier.in_features

        self.denseNet121.classifier = nn.Sequential(
            nn.Linear(features,return_size),
            nn.Sigmoid()
        )


    def forward(self,x):

        return self.denseNet121(x)