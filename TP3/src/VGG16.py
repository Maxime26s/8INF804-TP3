import logging
import torch.nn as nn
from torchvision import models


def initialize_vgg16_model(device):
    logging.info("Loading VGG16 pre-trained model")
    weights = models.VGG16_Weights.DEFAULT
    model = models.vgg16(weights=weights).to(device)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier[6] = nn.Linear(4096, 14)  # 14 classes
    return model.to(device)
