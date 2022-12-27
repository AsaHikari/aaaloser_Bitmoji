import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import shutil
from sklearn.metrics import f1_score, precision_score, recall_score
import os
###显卡设置
os.environ["CUDA_VISIBLE_DEVICES"] = "0"



def loadtraindata():
    # 路径
    path = "./data/train"
    trainset = torchvision.datasets.ImageFolder(path, transform=transforms.Compose([
        # 将图片缩放到指定大小（h,w）或者保持长宽比并缩放最短的边到int大小
        transforms.Resize((32, 32)),
        transforms.CenterCrop(32),
        transforms.ToTensor()])
                                                )
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
    return trainloader

def evaluate(model, data, device):
    val_targ = []
    val_pred = []
    with torch.no_grad():
        for (data_x, data_y) in data:
            data_x = data_x.to(device)
            data_y = data_y.to(device)
            logits = model(data_x)
            _, indices = torch.max(logits.to('cpu'), dim=1)
            val_pred.extend(indices)
            val_targ.extend(data_y.to('cpu'))
        # print(val_pred,val_targ)
        val_f1 = f1_score(val_targ, val_pred)
        val_recall = recall_score(val_targ, val_pred)
        val_precision = precision_score(val_targ, val_pred)

    return val_precision, val_recall, val_f1

# 定义网络，继承torch.nn.Module
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def trainandsave():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # torchvision.models.googlenet
    model = torchvision.models.resnet50(pretrained=True).to(device) # or pretrained=False
    # model = Net().to(device)
    trainloader = loadtraindata()
    # 神经网络结构

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    criterion = nn.CrossEntropyLoss()

    for epoch in range(50):
        # 每个epoch要训练所有的图片，每训练完成200张便打印一下训练的效果（loss值）
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # enumerate是python的内置函数，既获得索引也获得数据
            # data是从enumerate返回的data，包含数据和标签信息，分别赋值给inputs和labels
            inputs, labels = data

            inputs, labels = Variable(inputs).to(device), Variable(labels).to(device)
            optimizer.zero_grad()

            outputs = model(inputs).to(device)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 200 == 199:
                P, R, F1 = evaluate(model,trainloader,device)
                print('[{:d},{:5d}] loss:{:.4f},Precission:{:.4f},Recall:{:.4f},F1:{:.4f}'.format(epoch + 1, i + 1, running_loss / 200, P, R, F1))
                running_loss = 0.0

        # 保存神经网络
        netScript = torch.jit.script(model)
        # 保存整个神经网络的结构和模型参数
        torch.jit.save(netScript, './model_save/epoch{:d}_Lenet.pt'.format(epoch + 1))

    print('Finished Training')

    # torch.jit.save(net.state_dict(), 'net_params.pt')


if __name__=="__main__":
    shutil.rmtree('./model_save')
    os.mkdir('./model_save')
    trainandsave()