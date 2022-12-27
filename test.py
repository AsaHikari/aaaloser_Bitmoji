import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dataload import Bitmoji

classes = ('1','-1')
mbatch_size = 5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

testset = Bitmoji(root=r'.', test=True)

def loadtestdata():
    path = "testimages"
    '''testset = torchvision.datasets.ImageFolder(path,
                                                transform=transforms.Compose([
                                                    transforms.Resize((32, 32)),
                                                    transforms.ToTensor()])
                                                )'''
    testloader = torch.utils.data.DataLoader(testset, batch_size=mbatch_size,
                                             shuffle=False, num_workers=2)
    return testloader

def reload_net():
    trainednet = torch.jit.load('./model_save/epoch48_Lenet.pt')
    return trainednet

def test():
    testloader = loadtestdata()
    net = reload_net()
    net.eval()
    result = []
    for data_x, _ in testloader:
        # print(data_y)
        data_x = data_x.to(device)
        out = net(data_x)
        _, indices = torch.max(out, dim=1)
        result.extend(indices.data.cpu().numpy())

    result = [np.array(1) if a == 1 else -1 for a in result]
    ind = list(range(3000, 4084))
    data = {'image_id': ind, 'is_males': result}
    df = pd.DataFrame(data)
    df.to_csv(r'./out.csv', index=False)


if __name__=="__main__":
    test()