import os
import argparse

import cv2
import numpy as np

import torch
import torchvision.transforms as T
# print('PyTorch version: {}.'.format(torch.__version__))

from model import SimpleCLS
# init net
net = SimpleCLS(phase='test')
net = net.eval()
# load weight
state_dict = torch.load('./weights/face_binary_cls.pth')
net.load_state_dict(state_dict)


parser = argparse.ArgumentParser('Face binary classification')
parser.add_argument('--img', default='samples/face.jpg', type=str)
args = parser.parse_args()

# read image
img = cv2.imread(args.img)
with torch.no_grad():
    out = net(T.ToTensor()(img).unsqueeze(0))
    conf_scores = out.squeeze(0).data.numpy()
print('bg score: {:.6f}, face score: {:.6f}.'.format(conf_scores[0], conf_scores[1]))
