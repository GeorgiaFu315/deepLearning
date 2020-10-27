import torch.nn as nn
import torch
import math
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


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, n_class=2300, input_size=32, width_mult=1.):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 4096
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 1],
            [6, 32, 3, 1],
            [6, 64, 4, 1],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        # input_channel = make_divisible(input_channel * width_mult)  # first channel is always 32!
        self.last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 1)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = make_divisible(c * width_mult) if t > 1 else c
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier_layer1 = nn.Linear(self.last_channel, 4096)
        self.bnorm1 = nn.BatchNorm1d(4096)
        self.classifier_layer2 = nn.Linear(4096, 4096)
        self.bnorm2 = nn.BatchNorm1d(4096)
        self.classifier_layer3 = nn.Linear(4096, 2300)


        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.bnorm1(self.classifier_layer1(x))
        x = self.bnorm2(self.classifier_layer2(x))
        x = self.classifier_layer3(x)
#         embed = self.classifier_verify(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

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

def parse_verify_data(datadir):
    img_pair = []
    with open('test_trials_verification_student.txt', 'r') as f:
        for line in f:
            img_pair.append(line.rstrip())
    # print(img_pair[0])

    img1_list = []
    img2_list = []
    for root, directories, filenames in os.walk(datadir):  # root: median/1
        print('#of pair', len(img_pair))
        for pair in img_pair:
            # print(pair.split()[0])
            filei_img1 = os.path.join(root, pair.split()[0])
            filei_img2 = os.path.join(root, pair.split()[1])
            img1_list.append(filei_img1)
            img2_list.append(filei_img2)
    return np.array(img1_list), np.array(img2_list)
    del img_list
    del ID_list

class ImageDataset_verify(Dataset):
    def __init__(self, img1_list, img2_list):
        self.img1_list = img1_list
        self.img2_list = img2_list

    def __len__(self):
        return len(self.img1_list)

    def __getitem__(self, index):
        img1_object = Image.open(self.img1_list[index])
        img1 = torchvision.transforms.ToTensor()(img1_object)
        img2_object = Image.open(self.img2_list[index])
        img2 = torchvision.transforms.ToTensor()(img2_object)
        return img1, img2

def test_verify(model, test_verify_dataloader):
    model.eval()
    pred_dis=torch.tensor([])
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    with torch.no_grad():
        for batch_num,(img1, img2) in enumerate(test_verify_dataloader):
            img1,img2 = img1.to(device),img2.to(device)
            embed_img1,embed_img2 = model(img1)[0],model(img2)[0]
            cos_output = cos(embed_img1, embed_img2)
            pred_dis=torch.cat((pred_dis,cos_output.detach().cpu()))
            if(len(pred_dis)%1024==0):
                print(len(pred_dis))
    # write file
    with open('predict_verify_number.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, lineterminator='\n')
        for index, value in enumerate(pred_dis):
            writer.writerow([index, value.numpy()])
    del img1
    del img2

def learn(model, data_loader, test_loader, scheduler):
    for epoch in range(numEpochs):
        start_time = time.time()
        train(epoch,model, data_loader,scheduler)
        train_loss, train_acc = val_classify(model, data_loader)
        val_loss, val_acc = val_classify(model, test_loader)
        print('Train Loss: {:.4f}\tTrain Accuracy: {:.4f}\tVal Loss: {:.4f}\tVal Accuracy: {:.4f}'.
                      format(train_loss, train_acc, val_loss, val_acc))
        torch.save(model.state_dict(), './modelHyperparameters_verify_model_new.pt')
        end_time = time.time()
        print('Epoch: {}\tTime: {}'.format(epoch+1,end_time-start_time))
        #save model
        # torch.save(model, './modelHyperparameters_verify_mobileNetv3.pt')
        torch.save(model.state_dict(), './modelHyperparameters_verify_mobileNetv3_model_new.pt')

def train(epoch,model, data_loader,scheduler):
    model.train()
    avg_loss = 0.0
    for batch_num, (feats, labels) in enumerate(data_loader):
        feats, labels = feats.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(feats)

        loss = criterion(outputs, labels.long())
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()

        if batch_num % 100 == 99:
            print('Epoch: {}\tBatch: {}\tAvg-Loss: {:.4f}'.format(epoch+1, batch_num+1, avg_loss/100))
            train_loss_100=avg_loss/100
            scheduler.step(train_loss_100)
            avg_loss = 0.0

        torch.cuda.empty_cache()
        del feats
        del labels
        del loss


def val_classify(model, test_loader):
    with torch.no_grad():
        model.eval()
        test_loss = []
        accuracy = 0
        total = 0

        for batch_num, (feats, labels) in enumerate(test_loader):
            feats, labels = feats.to(device), labels.to(device)
#             print(feats.shape)
            outputs = model(feats)

            _, pred_labels = torch.max(F.softmax(outputs, dim=1), 1)
            pred_labels = pred_labels.view(-1)
#             print(pred_labels)
            loss = criterion(outputs, labels.long())

            accuracy += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
            test_loss.extend([loss.item()]*feats.size()[0])
            del feats
            del labels

        model.train()
        return np.mean(test_loss), accuracy/total
        del loss

if __name__=="__main__":
    train_augmentation = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),
                                                    torchvision.transforms.ToTensor()])
    train_augmentation2 = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])


    #load train data
    print('Load Train Data...')
    train_dataset = torchvision.datasets.ImageFolder(root='train_data/medium',transform=train_augmentation)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=256,shuffle=True, num_workers=8)

    #load dev data
    print('Load Dev Data...')
    dev_dataset = torchvision.datasets.ImageFolder(root='validation_classification/medium',transform=train_augmentation2)
    dev_dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=256,shuffle=True, num_workers=8)

    #train
    numEpochs = 20
    num_feats = 3
    learningRate = 1e-2
    weightDecay = 5e-4
    # num_classes = len(train_dataset.classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #load model
    network = MobileNetV2()

    PATH='./modelHyperparameters_verify_mobileNetv3_model.pt'
    # network.load_state_dict(torch.load(PATH))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(network.parameters(), lr=learningRate, weight_decay=weightDecay, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, threshold=0.01, patience=3)
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.9)#reduce plateor#reduce lr pla,trasforms-random horizontal flip,

    network.train()
    network.to(device)
    print('start training....')
    learn(network, train_dataloader, dev_dataloader, scheduler)

    #load test data
    print("Load Test Data...")
    test_img_list, test_ID_list = parse_data('test_classification/medium')
    test_dataset = ImageDataset(test_img_list, test_ID_list)
    test_dataloader = DataLoader(test_dataset, batch_size=4600, shuffle=False, num_workers=1, drop_last=False)

    #test-verification
    img1_list,img2_list=parse_verify_data('./test_verification')
    test_verify_dataset = ImageDataset_verify(img1_list, img2_list)
    test_verify_dataloader = DataLoader(test_verify_dataset, batch_size=256, shuffle=False, num_workers=1, drop_last=False)
    test_verify(network, test_verify_dataloader)
