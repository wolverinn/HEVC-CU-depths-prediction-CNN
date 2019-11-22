import pickle
import os
from PIL import Image
import math
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 让torch判断是否使用GPU
model = ConvNet2().to(DEVICE)
model.load_state_dict(torch.load('./hevc_encoder_model.pt',map_location='cpu'))
print("loaded model from drive")

def process_rdcost(ctu_x,ctu_y):
    current_frame = 0
    current_ctu = 0
    depth_cost = {}
    with open("rdcost.txt",'r',encoding='utf-8') as f:
        for i,line in enumerate(f):
            if "frame" in line:
                current_frame = line.split(":")[-1].strip('\n')
                depth_cost[current_frame] = {}
            elif "ctu" in line:
                current_ctu = line.split(":")[-1].strip('\n')
                depth_cost[current_frame][current_ctu] = {"0":[],"1":[],"2":[],"3":[]}
            else:
                depth = line.split(":")[1]
                depth_cost[current_frame][current_ctu][depth].append(float(line.split(":")[2].strip('\n')))
    os.remove("./rdcost.txt")
    # process depth_cost to fill the missing costs
    for frame_number in depth_cost.keys():
        for ctu_number in depth_cost[frame_number].keys():
            if len(depth_cost[frame_number][ctu_number]["1"]) == 4:
                pass
            else:
                if (int(ctu_number) + 1) % ctu_x == 0 and ((int(ctu_number) + 1) // ctu_x) == ctu_y:
                    depth_cost[frame_number][ctu_number]["0"] = sum(depth_cost[frame_number][ctu_number]["1"])
                    depth_cost[frame_number][ctu_number]["1"] += [0,0,0]
                    depth_cost[frame_number][ctu_number]["2"] += list(np.zeros(12))
                    depth_cost[frame_number][ctu_number]["3"] += list(np.zeros(96))
                elif (int(ctu_number) + 1) % ctu_x == 0:
                    depth_cost[frame_number][ctu_number]["0"] = sum(depth_cost[frame_number][ctu_number]["1"])
                    depth_cost[frame_number][ctu_number]["1"].insert(1,0.0)
                    depth_cost[frame_number][ctu_number]["1"].append(0.0)
                    depth_cost[frame_number][ctu_number]["2"] = depth_cost[frame_number][ctu_number]["2"][0:4] + list(np.zeros(4)) + depth_cost[frame_number][ctu_number]["2"][8:12] + list(np.zeros(4))
                    depth_cost[frame_number][ctu_number]["3"] = depth_cost[frame_number][ctu_number]["3"][0:32] + list(np.zeros(32)) + depth_cost[frame_number][ctu_number]["3"][64:96] + list(np.zeros(32))
                else:
                    depth_cost[frame_number][ctu_number]["0"] = sum(depth_cost[frame_number][ctu_number]["1"])
                    depth_cost[frame_number][ctu_number]["1"] += [0,0]
                    depth_cost[frame_number][ctu_number]["2"] += list(np.zeros(8))
                    depth_cost[frame_number][ctu_number]["3"] += list(np.zeros(64))
    return depth_cost

def get_ctu_cost(cost_dict,label_list): # cost_dict = depth_cost[frame_num][ctu_num]
    label_index = [
        [0,0], # where to find rd-cost of depth-1 and depth-2. depth-3-start = depth-2*8, depth-3-end = start+7
        [0,1],
        [1,4],
        [1,5],
        [0,2],
        [0,3],
        [1,6],
        [1,7],
        [2,8],
        [2,9],
        [3,12],
        [3,13],
        [2,10],
        [2,11],
        [3,14],
        [3,15]
    ]
    ctu_cost = 0
    if 0 in label_list:
        return cost_dict["0"]
    for i,depth in enumerate(label_list):
        if depth == 1:
            try:
                ctu_cost += cost_dict["1"][label_index[i][0]]/4
            except:
                pass
        elif depth == 2:
            try:
                ctu_cost += cost_dict["2"][label_index[i][1]]
            except:
                pass
        else:
            try:
                temp_depth3_cost = cost_dict["3"][label_index[i][1]*8:(label_index[i][1]*8+8)]
                ctu_cost += np.min(temp_depth3_cost[0:2])+np.min(temp_depth3_cost[2:4])+np.min(temp_depth3_cost[4:6])+np.min(temp_depth3_cost[6:8])
            except:
                pass
    return ctu_cost

YUV_FILE_PATH = ".\\test_cost"
WORKSPACE_PATH = os.getcwd()
CtuInfo_FILENAME = "ctu_depth.txt"

def gen_cfg(yuv_filename):
    FrameRate = yuv_filename.split('_')[2].strip(".yuv")
    SourceWidth = yuv_filename.split('_')[1].split('x')[0]
    SourceHeight = yuv_filename.split('_')[1].split('x')[1]
    with open('.\\config\\bitstream.cfg','w') as f:
        f.write("InputFile : {}\\{}\n".format(YUV_FILE_PATH,yuv_filename))
        f.write("InputBitDepth : 8\n")
        f.write("InputChromaFormat : 420\n")
        f.write("FrameRate : {}\n".format(FrameRate))
        f.write("FrameSkip : 0\n")
        f.write("SourceWidth : {}\n".format(SourceWidth))
        f.write("SourceHeight : {}\n".format(SourceHeight))
        f.write("FramesToBeEncoded : 10000\n")
        f.write("Level : 3.1")

def dump_ctu_file(video_number,frame_number):
    frame_detected = 0
    ctu_number = "0"
    temp_ctu = []
    video_dict = {}
    video_dict[frame_number] = {}
    with open(CtuInfo_FILENAME,'r') as f:
        for i,line in enumerate(f):
            if frame_detected == 0:
                if "frame" in line:
                    current_frame = line.split(':')[1]
                    if int(frame_number) == int(current_frame):
                        frame_detected = 1
            elif "frame" in line:
                break
            elif "ctu" in line:
                temp_ctu = []
                ctu_number = int(line.split(':')[1])
                line_count = 0
                video_dict[frame_number][str(ctu_number)] = []
            else:
                line_depths = line.split(' ')
                if line_count % 4 == 0:
                    for index in range(4):
                        temp_ctu.append(int(line_depths[4*index]))
                        video_dict[frame_number][str(ctu_number)] = temp_ctu
                line_count += 1
    if video_dict[frame_number] == {}:
        video_dict.pop(frame_number)
    return video_dict

def predict_label(frame_number):
    img = Image.open("./temp-frames/{}.jpg".format(str(int(frame_number)+1)))
    img_width, img_height = img.size
    ctu_x = math.ceil(img_width / 64)
    ctu_y = math.ceil(img_height / 64)
    ctu_numbers = ctu_x * ctu_y
    label_dict = {}
    label_dict[frame_number] = {}
    with torch.no_grad():
        for i in range(ctu_numbers):
            label = []
            for n in range(16):
                label.append(str(n))
            img_row = i // math.ceil(img_width / 64)
            img_colonm = i % math.ceil(img_width / 64)
            for layer2 in range(4):
                start_pixel_x = img_colonm * 64 + (layer2 % 2)*32
                start_pixel_y = img_row * 64 + (layer2 // 2)*32
                cropped_img32 = img.crop((start_pixel_x, start_pixel_y, start_pixel_x + 32, start_pixel_y + 32))
                cropped_img64 = img.crop((img_colonm * 64, img_row * 64, img_colonm * 64 + 64, img_row * 64 + 64))
                data32 = transforms.ToTensor()(cropped_img32).unsqueeze(0)
                data64 = transforms.ToTensor()(cropped_img64).unsqueeze(0)
                cropped_img32.close()
                cropped_img64.close()
                data32 = data32.to(DEVICE)
                data64 = data64.to(DEVICE)
                output = model(data32,data64)
                pred = str(int(torch.argmax(output[0,0:4]))) + str(int(torch.argmax(output[0,4:8]))) + str(int(torch.argmax(output[0,8:12]))) + str(int(torch.argmax(output[0,12:16])))
                if "0" in pred and pred != "0000":
                    pred = pred.replace("0","1")
                if "1" in pred and pred != "1111":
                    pred = pred.replace("1","2")
                if layer2 == 0:
                    label[0],label[1],label[4],label[5] = pred[0],pred[1],pred[2],pred[3]
                elif layer2 == 1:
                    if pred == "0000" and label[0] != "0":
                        pred = "1111"
                    label[2],label[3],label[6],label[7] = pred[0],pred[1],pred[2],pred[3]
                elif layer2 == 2:
                    if pred == "0000" and label[2] != "0":
                        pred = "1111"
                    label[8],label[9],label[12],label[13] = pred[0],pred[1],pred[2],pred[3]
                else:
                    if pred == "0000" and label[8] != "0":
                        pred = "1111"
                    label[10],label[11],label[14],label[15] = pred[0],pred[1],pred[2],pred[3]
            label = [int(x) for x in label]
            label_dict[frame_number][str(i)] = label
    img.close()
    os.remove("./temp-frames/{}.jpg".format(str(int(frame_number)+1)))
    return label_dict,ctu_x,ctu_y

encoding_cmd = "TAppEncoder.exe -c .\\config\\encoder_intra_main.cfg -c .\\config\\bitstream.cfg"
try:
    os.mkdir("temp-frames")
except:
    pass
for i,yuv_filename in enumerate(os.listdir(YUV_FILE_PATH)):
    if i == 1:
        break
    gen_cfg(yuv_filename)
    os.system(encoding_cmd)
    # ffmpeg -video_size 352x288 -r 20 -pixel_format yuv420p -i E:\HM\trunk\workspace\yuv-resources\train\paris_352x288_20.yuv E:\HM\trunk\workspace\temp-frames\v_0_%d_.jpg
    gen_frames_cmd = "ffmpeg -video_size {} -r {} -pixel_format yuv420p -i {}\\{} {}\\temp-frames\\%d.jpg".format(yuv_filename.split('_')[1],yuv_filename.split('_')[2].strip(".yuv"),YUV_FILE_PATH,yuv_filename,WORKSPACE_PATH)
    os.system(gen_frames_cmd)
    print("processing yuv file: {}".format(yuv_filename))
    total_frames = len(os.listdir("./temp-frames"))
    # total_frames = 2
    predict_dict = {}
    original_dict = {}
    ctu_x,ctu_y = 0,0
    for frame_number in range(total_frames):
        temp,ctu_x,ctu_y = predict_label(str(frame_number))
        predict_dict[str(frame_number)] = temp[str(frame_number)]
        original_dict[str(frame_number)] = dump_ctu_file(str(i), str(frame_number))[str(frame_number)]
    os.remove(CtuInfo_FILENAME)
    # start calculate rd-cost
    depth_cost = process_rdcost(ctu_x,ctu_y)
    # calculate original rd-cost
    ori_cost = 0
    pred_cost = 0
    for frame_number in range(total_frames):
        for ctu_number in original_dict["0"].keys():
            temp_ori_cost = get_ctu_cost(depth_cost[str(frame_number)][ctu_number],original_dict[str(frame_number)][ctu_number])
            ori_cost += temp_ori_cost
            temp_pred_cost = get_ctu_cost(depth_cost[str(frame_number)][ctu_number],predict_dict[str(frame_number)][ctu_number])
            pred_cost += temp_pred_cost
    print("HEVC encoder RD-cost: {}".format(ori_cost))
    print("CNN model RD-cost: {}".format(pred_cost))
    print("cost difference: {}%".format(str((pred_cost-ori_cost)/ori_cost/100)))