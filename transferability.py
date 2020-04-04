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
parser.add_argument("-S", "--image", help="if save confusion matrix or not", default = True, dest="save")

args = parser.parse_args()
input_path = args.input
save_images = args.save

if __name__ == "__main__":
    models = ["vgg16_bn", "resnet50", "mobilenet_v2", "densenet161"]
    modes = ["FGSM", "PGD"]
    normal_accuracy_list = []
    all_accuracy = []
    for mode in modes:
        sub_accuracy = np.zeros((4,4))
        for model_id ,try_model in enumerate(models):
            print("==========================================")
            print("=====" + mode + " | " + try_model + "=====")
            print("==========================================")
            model = select_model(try_model, pretrained = True)
            save_name = "adv_imgs_" + mode + "_" + try_model
            accuracy, normal_accuracy = adversarial_read_gen_save(image_path = input_path, model = model , eps = 0.150, alpha=2/255, attack_type=mode, adv_output = save_name)
            normal_accuracy_list.append(normal_accuracy)
            sub_accuracy[model_id,model_id] = accuracy
            for transfer_model_idx, transfer_model in enumerate(models):
                if transfer_model == try_model:
                    continue
                else:
                    print("== Transfer test on" + transfer_model + " ==")
                    accuracy = transfer_attack(save_name, transfer_model)
                    sub_accuracy[model_id, transfer_model_idx] = accuracy
        all_accuracy.append(sub_accuracy)
        print("=======================")
        print("Each normal accuracy")
        print(normal_accuracy_list[:4])

    if save_images:
        for idx, mode in enumerate(modes):
            df_cm = pd.DataFrame(all_accuracy[idx], index = [i for i in models],
                            columns = [i for i in models])
            plt.figure(figsize = (10,7))

            plt.title(mode, size = 20)
            
            ax = sns.heatmap(df_cm, annot=True)
            ax.set(xlabel='To', ylabel='From')
            plt.savefig(mode+".png")