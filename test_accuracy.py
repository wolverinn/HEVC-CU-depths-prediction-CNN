import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data as data
from torchvision import datasets, transforms
import os
import pickle
import numpy as np
from PIL import Image
import time
import math

LOAD_DIR = "."
BATCH_SIZE=1024
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 让torch判断是否使用GPU
if torch.cuda.is_available():
  print("cuda is available")
else:
  print("cuda unavailable")

class ConvNet2(nn.Module):
    def __init__(self):
        super().__init__()
        # (3,32,32)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,16,5,padding=2),
            nn.BatchNorm2d(16,affine=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
            ) # (16,16,16)
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,64,3,padding=1),
            nn.BatchNorm2d(64,affine=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
            ) # (64,8,8)
        self.conv3 = nn.Sequential(
            nn.Conv2d(64,128,3,padding=1),
            nn.BatchNorm2d(128,affine=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
            ) # (128,4,4)
        self.fc1 = nn.Sequential(nn.Linear(128*4*4,256),nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(256,64),nn.ReLU())
        self.fc3 = nn.Linear(64,16)
        self.conv64 = nn.Sequential(
            nn.Conv2d(3,16,5,padding=2),
            nn.BatchNorm2d(16,affine=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4)
            ) # (16,16,16) -> (64,16,16)
        # self.dropout = nn.Dropout(0.25)
    def forward(self,x32,x64):
        in_size = x32.size(0)
        out = torch.cat([self.conv1(x32),self.conv64(x64)],dim=1)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out.view(in_size,-1) # 扁平化flat然后传入全连接层
        out = self.fc1(out)
        # out = self.dropout(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out

transform = transforms.Compose([transforms.ToTensor()])

def from_ctufile(load_type,video_number,frame_number,ctu_number,layer2):
    # https://pytorch-cn.readthedocs.io/zh/latest/package_references/Tensor/
    ctu_file = "{}/dataset/pkl/{}/v_{}.pkl".format(LOAD_DIR,load_type,video_number)
    f_pkl = open(ctu_file,'rb')
    video_dict = pickle.load(f_pkl)
    f_pkl.close()
    ctu_info = video_dict[frame_number][ctu_number]
    if layer2 == 0:
        label_list = [ctu_info[0],ctu_info[1],ctu_info[4],ctu_info[5]]
    elif layer2 == 1:
        label_list = [ctu_info[2],ctu_info[3],ctu_info[6],ctu_info[7]]
    elif layer2 == 2:
        label_list = [ctu_info[8],ctu_info[9],ctu_info[12],ctu_info[13]]
    elif layer2 == 3:
        label_list = [ctu_info[10],ctu_info[11],ctu_info[14],ctu_info[15]]
    else:
        print("layer2 loading error!!!")
    label = torch.tensor(label_list)
#     label = one_hot_label(label_list)
    return label

class ImageSet(data.Dataset):
    def __init__(self,root):
        # 所有图片的绝对路径
        self.img_files = []
        self.root = root
        for img in os.listdir(root):
            ctu_numbers_per_frame = img.split('_')[3]
            for ctu_number in range(int(ctu_numbers_per_frame)):
                for layer2 in range(4):
                    self.img_files.append((img,ctu_number,layer2))
        self.transforms=transform

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.root,self.img_files[index][0]))
        video_number = self.img_files[index][0].split('_')[1]
        frame_number = self.img_files[index][0].split('_')[2]
        ctu_number = self.img_files[index][1]
        layer2 = self.img_files[index][2]
        img_width, _ = img.size
        img_row = ctu_number // math.ceil(img_width / 64)
        img_colonm = ctu_number % math.ceil(img_width / 64)
        start_pixel_x = img_colonm * 64 + (layer2 % 2)*32
        start_pixel_y = img_row * 64 + (layer2 // 2)*32
        cropped_img32 = img.crop((start_pixel_x, start_pixel_y, start_pixel_x + 32, start_pixel_y + 32)) # 依次对抽取到的帧进行裁剪
        cropped_img64 = img.crop((img_colonm * 64, img_row * 64, img_colonm * 64 + 64, img_row * 64 + 64))
        img.close()
        if "train" in self.root:
            load_type = "train"
        elif "validation" in self.root:
            load_type = "validation"
        elif "test" in self.root:
            load_type = "test"
        else:
            print("load type error!!!")
        img_data32 = self.transforms(cropped_img32)
        img_data64 = self.transforms(cropped_img64)
        cropped_img32.close()
        cropped_img64.close()
        label = from_ctufile(load_type,video_number,frame_number,str(ctu_number),layer2)
        return img_data32,img_data64,label,layer2

    def __len__(self):
        return len(self.img_files)

test_loader = data.DataLoader(ImageSet("./dataset/img/test/"),batch_size=BATCH_SIZE,shuffle=False)
model = ConvNet2().to(DEVICE)
model.load_state_dict(torch.load('{}/hevc_encoder_model.pt'.format(LOAD_DIR)))
print("loaded model from drive")
print(model)
criterion = nn.CrossEntropyLoss()

def test(model, device, test_loader):
    model.load_state_dict(torch.load('hevc_encoder_model.pt'))
    model.eval()
    test_loss = 0
    correct = 0
    label = []
    for i in range(16):
        label.append(str(i))
    with torch.no_grad():
        for img_data32,img_data64, target,layer2 in test_loader:
            img_data32,img_data64, target = img_data32.to(device),img_data64.to(device), target.to(device)
            output = model(img_data32,img_data64)
            test_loss += criterion(output[:,0:4], target[:,0]).item()+criterion(output[:,4:8], target[:,1]).item()+criterion(output[:,8:12], target[:,2]).item()+criterion(output[:,12:16], target[:,3]).item() # 将一批的损失相加
            for i,single_pred in enumerate(output):
                pred_0 = torch.argmax(single_pred[0:4])
                pred_1 = torch.argmax(single_pred[4:8])
                pred_2 = torch.argmax(single_pred[8:12])
                pred_3 = torch.argmax(single_pred[12:16])
                pred = str(int(pred_0)) + str(int(pred_1)) + str(int(pred_2)) + str(int(pred_3))
                if "0" in pred and pred != "0000":
                    pred = pred.replace("0","1")
                if "1" in pred and pred != "1111":
                    pred = pred.replace("1","2")
                if int(layer2[i]) == 0:
                    label[0],label[1],label[4],label[5] = pred[0],pred[1],pred[2],pred[3]
                elif int(layer2[i]) == 1:
                    if pred == "0000" and label[0] != "0":
                        pred = "1111"
                    label[2],label[3],label[6],label[7] = pred[0],pred[1],pred[2],pred[3]
                elif int(layer2[i]) == 2:
                    if pred == "0000" and label[2] != "0":
                        pred = "1111"
                    label[8],label[9],label[12],label[13] = pred[0],pred[1],pred[2],pred[3]
                else:
                    if pred == "0000" and label[8] != "0":
                        pred = "1111"
                    label[10],label[11],label[14],label[15] = pred[0],pred[1],pred[2],pred[3]
                target_0 = int(target[i,0])
                target_1 = int(target[i,1])
                target_2 = int(target[i,2])
                target_3 = int(target[i,3])
                if str(pred[0]) == str(target_0):
                    correct += 1
                if str(pred[1]) == str(target_1):
                    correct += 1
                if str(pred[2]) == str(target_2):
                    correct += 1
                if str(pred[3]) == str(target_3):
                    correct += 1
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(test_loss, correct, len(test_loader.dataset)*4,
        25. * correct / len(test_loader.dataset)))

test(model, DEVICE, test_loader)