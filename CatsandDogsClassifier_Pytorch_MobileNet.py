import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from torchinfo import summary
import os

from pytorchcv import train, display_dataset, train_long, load_cats_dogs_dataset, validate, common_transform




dataset, train_loader, test_loader = load_cats_dogs_dataset()


model = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True)
model.eval()
print(model)



sample_image = dataset[0][0].unsqueeze(0)
res = model(sample_image)
print(res[0].argmax())


#Note that the number of parameters in MobileNet and full-scale ResNet model differ significantly. 
# In some ways, MobileNet is more compact that VGG model family, which is less accurate. However,
#  reduction in the number of parameters naturally leads to some drop in the model accuracy.


#freeze all parameters of the model:

for x in model.parameters():
    x.requires_grad = False


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.classifier = nn.Linear(1280,2)
model = model.to(device)
summary(model,input_size=(1,3,244,244))


train_long(model,train_loader,test_loader,loss_fn=torch.nn.CrossEntropyLoss(),epochs=1,print_freq=90)

