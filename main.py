import os
import shutil
import cv2
import numpy as np
import pickle
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm as pbar
from cifar10_models import *
from utils import *
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from skimage.util import random_noise
import time
from argparse import ArgumentParser

cifar10_label_dict = {"airplane":0, "automobile":1, "bird":2, "cat":3, "deer":4, 
                      "dog":5, "frog":6, "horse":7, "ship":8, "truck":9}
cifar10_label_dict_reverse = {v: k for k, v in cifar10_label_dict.items()}

parser = ArgumentParser()
parser.add_argument("-I", "--input", help="the location of cifar-10_eval folder", default = "cifar-10_eval", dest="input")
parser.add_argument("-M", "--model", help="the model to generate adversarial examples: vgg16_bn, resnet50, mobilenet_v2, densenet161", default = "resnet50", dest="model")
parser.add_argument("-T", "--mode", help="the mode to generate adversarial examples: PGD or FGSM", default = "PGD", dest="mode")
parser.add_argument("-O", "--output", help="the location to save the output adversarial examples", default = "adv_imgs", dest="output")

args = parser.parse_args()
input_path = args.input
model = args.model
mode = args.mode
output = args.output

if __name__ == "__main__":
    model = select_model(model, pretrained = True)
    #model.load_state_dict(torch.load("cifar10_models/state_dicts/densenet161_gaussian.pt"))
    use_cuda=True
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
    accuracy, normal_accuracy = adversarial_read_gen_save(image_path = input_path, model = model , eps = 0.150, alpha=2/255, attack_type=mode, adv_output = output)