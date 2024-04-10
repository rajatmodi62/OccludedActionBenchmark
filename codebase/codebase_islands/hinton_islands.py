import sys
import os
import glob
# import utils
import torch
import random
import cv2
import time
import argparse
import datetime
import numpy as np
import os.path as osp
from pathlib import Path
from shutil import copy2
from caps_mvitv2 import CapsNet
from slowfast.utils.parser import load_config
from slowfast.utils.parser import parse_args as slowfast_parse_args
from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats
from slowfast.models import build_model
import slowfast.models.optimizer as optim
import slowfast.utils.checkpoint as cu
import slowfast.utils.logging as logging
from einops import reduce, rearrange, repeat
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import shutil as sh 
from pathlib import Path
import glob
import warnings
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore")

def read_video_as_tensor(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    n_frames= 0
    while(cap.isOpened()):
        if len(frames) == 1000:
            break
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32) / 255.0
            frames.append(frame)
        else:
            break
    
    # Release the VideoCapture object
    cap.release()
    
    # Stack the frames along the first axis to create a tensor
    frames = frames[-8:]
    video_tensor = np.stack(frames, axis=0)
    
    return video_tensor
def center_crop_video(video_tensor, crop_size=(224, 224)):
    # Get the dimensions of the video tensor
    num_frames, height, width, channels = video_tensor.shape
    
    # Calculate the starting and ending indices for cropping
    start_h = (height - crop_size[0]) // 2
    start_w = (width - crop_size[1]) // 2
    end_h = start_h + crop_size[0]
    end_w = start_w + crop_size[1]
    
    # Perform center cropping for each frame
    cropped_frames = []
    for frame in video_tensor:
        cropped_frame = frame[start_h:end_h, start_w:end_w, :]
        cropped_frames.append(cropped_frame)
    
    # Stack the cropped frames to form the cropped video tensor
    cropped_video_tensor = np.stack(cropped_frames, axis=0)
    
    return cropped_video_tensor

parser = argparse.ArgumentParser(description='evaluation')
parser.add_argument('--seed', type=int, default=47, help='seed for initializing training.')
parser.add_argument('--cfg', type=str, default='configs/Kinetics/mvits_with_weights.yaml')
n_frames = 8 # frames pumped through the transformer 
args = parser.parse_args()
saved_wts = Path('trained_weights/mvits_with_weights/best_model_train_loss_5.pth')

caps_args = slowfast_parse_args()
cfg = load_config(caps_args, args.cfg)
model = CapsNet(cfg).cuda()
model.load_previous_weights(saved_wts)
print("loaded")

image = cv2.imread('geoff.jpg')
image = cv2.resize(image, (224,224))
image = rearrange(image, 'h w c -> c h w').astype(np.float32)
image = repeat(image, 'c h w -> c t h w', t=n_frames)

video_tensor = torch.from_numpy(image).unsqueeze(0).cuda()

video_tensor = video_tensor.type(torch.cuda.FloatTensor)
video_tensor = video_tensor[:,:,:n_frames]
action = torch.tensor([0]).cuda()
empty_action = torch.tensor([0]).cuda()
feat = model.forward_island(video_tensor)
b = t = 0 # first batch, first frame
f = feat[b,:,t]
h,w =112,112

f = rearrange(f, 'c h w -> (h w) c').numpy()
print("fitting tsne")
#  rmodi: this guy is gotta take time: tsnecuda does not support > 2 components yet, maybe i will write something in future. 
f = TSNE(n_components=3, learning_rate='auto',

                init='random', perplexity=3).fit_transform(f)
print("done..")
f = (f-np.min(f))/(np.max(f)-np.min(f))
f = f*255
f = f.astype(np.uint8)
f = rearrange(f, '(h w) c ->  h w c', h=h, w=w)

to_save = '{}_{}.png'.format(b,t)
print("save island")
cv2.imwrite('island.png', f)
