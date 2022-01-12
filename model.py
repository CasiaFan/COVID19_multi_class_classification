import torch
import torch.nn as nn
from torchvision import models
from collections import OrderedDict

from torchvision.models import resnet
# from resnet import resnet50, resnet18, resnet101
# from densenet import densenet121, densenet161
# from inception_v3 import inception_v3

class LogitResnet(nn.Module):
    """ResNet architecture for extracting feature. Add an extra fc layer for extracting embedding."""
    def __init__(self, model_name, num_classes, embedding_dim=128, return_logit=False, use_pretrained=True, model_weight=None):
        """embeding """
        super(LogitResnet, self).__init__()
        if model_name == "resnet50":
            model = models.resnet50(pretrained=use_pretrained)
            # model = resnet50(pretrained=use_pretrained, model_weight=model_weight)
        elif model_name == "resnet101":
            model = models.resnet34(pretrained=use_pretrained)
            # model = resnet101(pretrained=use_pretrained, model_weight=model_weights) 
        elif model_name == "resnet18":
            model = models.resnet18(pretrained=use_pretrained)
            # model = resnet18(pretrained=use_pretrained, model_weight=model_weights)
        else:
            print("unknown resnet model")
            exit()
        num_features = model.fc.in_features
        self.return_logit = return_logit
        self.net = nn.Sequential(*list(model.children())[:-1])
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(num_features, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, num_classes)
    
    def forward(self, inputs):
        x = self.net(inputs)
        # x = self.avgpool(x)
        x = torch.flatten(x, 1)
        l = self.fc1(x)
        x = self.fc2(l)
        if self.return_logit:
            return [x, l]
        return [x]

class CifarResnet18(nn.Module):
    """use 3 resnet blocks (drop the last one)"""
    def __init__(self, num_classes, embedding_dim=128, return_logit=False, use_pretrained=True):
        super(CifarResnet18, self).__init__()
        model = models.resnet18(pretrained=use_pretrained)
        num_features = 128
        self.return_logit = return_logit
        self.net = nn.Sequential(*list(model.children())[:-4])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(num_features, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, num_classes)
    
    def forward(self, inputs):
        x = self.net(inputs)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        l = self.fc1(x)
        x = self.fc2(l)
        if self.return_logit:
            return [x, l]
        return [x]

class LogitDensenet(nn.Module):
    """return network logit"""
    def __init__(self, model_name, num_classes, embedding_dim=128, return_logit=False, use_pretrained=True, model_weight=None):
        super(LogitDensenet, self).__init__()
        if model_name == "densenet161":
            model = models.densenet161(pretrained=use_pretrained)
            # model = densenet161(pretrained=use_pretrained, model_weight=model_weight) 
        elif model_name == "densenet121":
            model = models.densenet121(pretrained=use_pretrained)
            # model = densenet121(pretrained=use_pretrained, model_weight=model_weight)
        else:
            print("unknown densenet structure")
        num_features = model.classifier.in_features
        self.return_logit = return_logit
        self.fc1 = nn.Linear(num_features, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, num_classes)
        self.net = nn.Sequential(*list(model.children())[:-1])
    
    def forward(self, inputs):
        x = self.net(inputs)
        x = nn.functional.relu(x, inplace=True)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        l = self.fc1(x)
        x = self.fc2(l)
        if self.return_logit:
            return [x, l]
        return [x]

# only for transfer learning 
class LogitInceptionV3(nn.Module):
    """ResNet architecture for extracting feature. Add an extra fc layer for extracting embedding."""
    def __init__(self, num_classes, use_pretrained=True, model_weight=None):
        """embeding """
        super(LogitInceptionV3, self).__init__()
        model = models.inception_v3(pretrained=use_pretrained, process=True)
        # model = inception_v3(pretrained=use_pretrained, model_weight=model_weight)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
        self.net = model
    
    def forward(self, inputs):
        x = self.net(inputs)
        return x


def C_net(model_name, num_classes, embedding_dim=128, use_pretrained=True, return_logit=False):
    """
    Get classifier network 
    """
    if model_name in ["resnet50", "resnet34", "resnet18"]:
        model = LogitResnet(model_name, num_classes, embedding_dim=embedding_dim, return_logit=return_logit, use_pretrained=use_pretrained)
    elif model_name in ["densenet121", "densenet161"]:
        model = LogitDensenet(model_name, num_classes, embedding_dim=embedding_dim, return_logit=return_logit, use_pretrained=use_pretrained)
    elif model_name == "cifar_resnet":
        model = CifarResnet18(num_classes, embedding_dim=embedding_dim, return_logit=return_logit, use_pretrained=use_pretrained)
    else:
        print("unknown model name!")
    return model

# load model
def load_from(net, weights, device, num_gpus=1):
    state_dict=torch.load(weights, map_location=torch.device(device))
    if num_gpus > 1:
        new_state_dict = OrderedDict()
        # remove 'module.' of dataparallel
        for k, v in state_dict.items():
            name = k[7:] 
            new_state_dict[name]=v
    else:
        new_state_dict = state_dict
    net.load_state_dict(new_state_dict)
    return net


if __name__ == "__main__":
    inputs = torch.rand(2, 3, 224, 224)
    inputs = torch.rand(2, 3, 112, 112) 
    model = C_net("cifar_resnet", 3, use_pretrained=False, return_logit=True) 
    # print(list(model.children())[:-1])
    # model = nn.Sequential(*list(model.children())[:-1])
    res = model(inputs)
    print(res[0].shape, res[1].shape)
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            print("\t", name, param.shape)
    print("-"*10)   
