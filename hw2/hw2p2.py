import os
import numpy as np
import torch
from PIL import Image
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import csv
import time

class ImageDataset(Dataset):
    def __init__(self, file_list, ID_list):
        self.file_list = file_list
        self.ID_list = ID_list
        self.n_class = len(list(set(ID_list)))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img = Image.open(self.file_list[index])
        img = torchvision.transforms.ToTensor()(img)
        ID = self.ID_list[index]
        return img, ID
def parse_data(datadir):
    filename_order = []
    with open ('test_order_classification.txt','r') as f:
        for line in f:
            filename_order.append(line.rstrip())
# print(filename_order)
    img_list = []
    ID_list = []
    for root, directories, filenames in os.walk(datadir):  #root: median/1
        print('#of file',len(filename_order))
        for filename in filename_order:
            filei = os.path.join(root, filename)
            img_list.append(filei)
            ID_list.append(filei.split('\\')[-1])
    print('{}\n{}'.format('#Images', len(img_list)))
    return np.array(img_list),np.array(ID_list)
    del img_list
    del ID_list

class Network(nn.Module):
    def __init__(self, num_feats, hidden_sizes, feat_dim=10):
        super(Network, self).__init__()

        self.layers = []
        self.layers.append(nn.Conv2d(in_channels=3, out_channels=64,kernel_size=5, stride=1,padding=2, bias=False))
        self.layers.append(nn.BatchNorm2d(64))
        self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(nn.MaxPool2d(kernel_size=3, stride=2,padding=1))
         #hidden layer2
        self.layers.append(nn.Conv2d(in_channels=64, out_channels=192, kernel_size=5, stride=1,padding=2, bias=False))
        self.layers.append(nn.BatchNorm2d(192))
        self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(nn.MaxPool2d(kernel_size=3, stride=2,padding=1))
          #hidden layer3
        self.layers.append(nn.Conv2d(in_channels=192,out_channels=384, kernel_size=3, stride=1,padding=1, bias=False))
        self.layers.append(nn.BatchNorm2d(384))
        self.layers.append(nn.ReLU(inplace=True))
        #hidden layer4
        self.layers.append(nn.Conv2d(in_channels=384, out_channels=256,kernel_size=3, stride=1,padding=1, bias=False))
        self.layers.append(nn.BatchNorm2d(256))
        self.layers.append(nn.ReLU(inplace=True))
        #hidden layer5
        self.layers.append(nn.Conv2d(in_channels=256,out_channels=256, kernel_size=3, stride=1,padding=1, bias=False))
        self.layers.append(nn.BatchNorm2d(256))
        self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(nn.MaxPool2d(kernel_size=3, stride=2,padding=1))

        self.layers = nn.Sequential(*self.layers)

        self.dp1 = nn.Dropout(p=0.2)
        self.linear_label1 = nn.Linear(4096, 4096, bias=False)
        self.bnorm1=nn.BatchNorm1d(4096)

        self.linear_label2 = nn.Linear(4096, 4096, bias=False)
        self.bnorm1=nn.BatchNorm1d(4096)
        self.linear_label3 = nn.Linear(4096, 2300, bias=False)

        # For creating the embedding to be passed into the Center Loss criterion
        self.linear_closs = nn.Linear(4096, feat_dim, bias=False)
        self.relu_closs = nn.ReLU(inplace=True)

    def forward(self, x, evalMode=False):
        output = x
        output = self.layers(output)

        output = output.view(output.shape[0], -1)
        output=self.dp1(output)
        output=F.relu(self.linear_label1(output))
        output=F.relu(self.linear_label2(output))
        output=self.bnorm1(output)
        label_output=self.linear_label3(output)
        label_output = label_output/torch.norm(self.linear_label3.weight, dim=1)

        # Create the feature embedding for the Center Loss
        closs_output = self.linear_closs(output)
        closs_output = self.relu_closs(closs_output)

        return closs_output, label_output

def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight.data)

def train(model, data_loader, test_loader, scheduler,task='Classification'):
    model.train()
    for epoch in range(numEpochs):
        scheduler.step()
        avg_loss = 0.0
        start_time = time.time()
        for batch_num, (feats, labels) in enumerate(data_loader):
            feats, labels = feats.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(feats)[1]

            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()

            if batch_num % 100 == 99:
                print('Epoch: {}\tBatch: {}\tAvg-Loss: {:.4f}'.format(epoch+1, batch_num+1, avg_loss/100))
                avg_loss = 0.0

            torch.cuda.empty_cache()
            del feats
            del labels
            del loss
        end_time = time.time()
        print('Epoch: {}\tTime: {}'.format(epoch+1,end_time-start_time))
        if task == 'Classification':
            val_loss, val_acc = val_classify(model, test_loader)
            train_loss, train_acc = val_classify(model, data_loader)
            print('Train Loss: {:.4f}\tTrain Accuracy: {:.4f}\tVal Loss: {:.4f}\tVal Accuracy: {:.4f}'.
                  format(train_loss, train_acc, val_loss, val_acc))
        else:
            test_verify(model, test_loader)
        #save model
        torch.save(model, './modelHyperparameters.pt')
        # torch.save(state, "C:\\Users\\georg\\Documents\\cmu\\cmu-course-2020spring\\deepLearning\\hw2\\hw2p2\\modelHyperparameters.pth")

def val_classify(model, test_loader):
    with torch.no_grad():
        model.eval()
        test_loss = []
        accuracy = 0
        total = 0

        for batch_num, (feats, labels) in enumerate(test_loader):
            feats, labels = feats.to(device), labels.to(device)
            outputs = model(feats)[1]

            _, pred_labels = torch.max(F.softmax(outputs, dim=1), 1)
            pred_labels = pred_labels.view(-1)

            loss = criterion(outputs, labels.long())

            accuracy += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
            test_loss.extend([loss.item()]*feats.size()[0])
            del feats
            del labels

        model.train()
        return np.mean(test_loss), accuracy/total
        del loss

def get_test_data(datadir):
    img_list = []
    ID_list = []
    for root, directories, filenames in os.walk(datadir):  #root: median/1
        for filename in filenames:
            if filename.endswith('.jpg'):
                filei = os.path.join(root, filename)
                img_list.append(filei)
                ID_list.append(filei.split('/')[-1])
    print('#Test Images', len(img_list))
    test_data = torch.stack([torchvision.transforms.ToTensor()(Image.open(img_list[idx])) for idx in range(len(img_list))])
    print(test_data.shape)
    return test_data
    del img_list
    del ID_list
    del test_data

def test_classify(model, test_dataloader):
#     print('test_data shape',test_dataloader.shape)
    # test_dataloader = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=1, drop_last=False)
    model.eval()
    with torch.no_grad():
        for batch_num, (img, ID) in enumerate(test_dataloader):
            img= img.to(device)
            outputs = model(img)[1]
            _, pred_labels = torch.max(F.softmax(outputs, dim=1), 1)
            pred_labels = pred_labels.view(-1)
    #define dictionary
    n_class=np.arange(2300)
    class_alha=sorted(n_class, key=str)
    dic_class=dict(zip(n_class,class_alha))

    #write file
    with open('predict_number.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, lineterminator='\n')
        for index,num in enumerate(pred_labels.detach().cpu().numpy()):
            writer.writerow([index,dic_class[num]])
    del img
    del ID

def test_verify(model, test_loader):
    raise NotImplementedError

if __name__=="__main__":
    #load train data
    print('Load Train Data...')
    train_dataset = torchvision.datasets.ImageFolder(root='train_data/medium',transform=torchvision.transforms.ToTensor())
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=256,shuffle=True, num_workers=8)

    # train_img_list, train_label_list, class_n = parse_data('train_data/medium')
    # train_dataset = ImageDataset(train_img_list, train_label_list)
    # train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=1, drop_last=False)
    #load dev data
    print('Load Dev Data...')
    dev_dataset = torchvision.datasets.ImageFolder(root='validation_classification/medium',transform=torchvision.transforms.ToTensor())
    dev_dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=256,shuffle=True, num_workers=8)
    # dev_img_list, dev_label_list, class_n = parse_data('validation_classification/medium')
    # dev_dataset = ImageDataset(dev_img_list, dev_label_list)
    # dev_dataloader = DataLoader(dev_dataset, batch_size=256, shuffle=True, num_workers=1, drop_last=False)
    #load test data
    print("Load Test Data...")
    test_img_list, test_ID_list = parse_data('test_classification/medium')
    test_dataset = ImageDataset(test_img_list, test_ID_list)
    test_dataloader = DataLoader(test_dataset, batch_size=4600, shuffle=False, num_workers=1, drop_last=False)

    # test_data= get_test_data('test_classification/medium')
    #check data load
    # train_data_item, train_data_label = trainset.__getitem__(417)#取第417个检查看看
    # print('data item shape: {}\t data item label: {}'.format(train_data_item.shape, train_data_label))

    #train
    numEpochs = 10
    num_feats = 3
    learningRate = 1e-6
    weightDecay = 5e-4
    hidden_sizes = [32, 64]
    # num_classes = len(train_dataset.classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #load model
    network = Network(num_feats,hidden_sizes)
#     network.apply(init_weights)

    PATH='./modelHyperparameters.pt'
    network =torch.load(PATH)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(network.parameters(), lr=learningRate, weight_decay=weightDecay, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.9)#reduce plateor#reduce lr pla,trasforms-random horizontal flip,

    network.train()
    network.to(device)
    print('start training....')
    train(network, train_dataloader, dev_dataloader,scheduler)

    #test
    test_classify(network, test_dataloader)
