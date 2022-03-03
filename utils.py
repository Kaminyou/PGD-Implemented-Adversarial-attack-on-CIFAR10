import copy
import os
import pickle
import shutil
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from skimage.util import random_noise
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm as pbar

from cifar10_models import *
from utils import *

cifar10_label_dict = {"airplane":0, "automobile":1, "bird":2, "cat":3, "deer":4, 
                      "dog":5, "frog":6, "horse":7, "ship":8, "truck":9}
cifar10_label_dict_reverse = {v: k for k, v in cifar10_label_dict.items()}
use_cuda=True
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
# Read eval image
def read_eval_image(target_path, label_dictionary = None):
    if label_dictionary == None:
        label_dictionary = {"airplane":0, "automobile":1, "bird":2, "cat":3, "deer":4, 
                            "dog":5, "frog":6, "horse":7, "ship":8, "truck":9}
    images_all = []
    category_all = []
    images_name_all = []
    for category in os.listdir(target_path):
        for img_name in os.listdir(target_path+"/"+category):
            image_path = os.path.join(target_path+"/"+category, img_name)
            image = cv2.imread(image_path)
            
            images_all.append(image)
            category_all.append(category)
            images_name_all.append(img_name)
    images_all = np.array(images_all)
    labels = np.array(list(map(lambda x: label_dictionary[x], category_all)))
    #print("Read images: ", len(images_all))
    #print("Output shape: ", images_all.shape)
    return images_all, labels, category_all, images_name_all

def unpickle(file):
    
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def To_noraml_RGB_image(img, channel_last = False):
    if not channel_last:
        img = img.transpose(1,2,0)
    img = img.clip(min = 0)
    img = img*255
    img = b = np.array(img, dtype=np.int16)
    img = img.clip(min = 0, max = 255)
    return img

def AdvRGB_dif(ori_img, adv_img, both_unmormal = True):
    if both_unmormal:
        ori_img = To_noraml_RGB_image(ori_img)
        adv_img = To_noraml_RGB_image(adv_img)
    difference = adv_img - ori_img
    print("Max: ", difference.max())
    print("Max: ", difference.min())
    return

def Show_image(img, channel_last = False):
    img = To_noraml_RGB_image(img, channel_last)
    plt.imshow(img)
    plt.show()

def select_model(model, pretrained = False, use_cuda = True):
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
    if model == "vgg16_bn":
        return vgg16_bn(pretrained = pretrained, device = device)
    elif model == "resnet50":
        return resnet50(pretrained = pretrained, device = device)
    elif model == "mobilenet_v2":
        return mobilenet_v2(pretrained = pretrained, device = device)
    elif model == "densenet161":
        return densenet161(pretrained = pretrained, device = device)
    
    elif model == "densenet161_gaussian":
        output_model = densenet161(pretrained = False, device = device)
        state_dict = torch.load('cifar10_models/state_dicts/'+model+'.pt', map_location=device)
        output_model.load_state_dict(state_dict)
        return output_model
    else:
        print("Unavailable model")
def generate_saving_image(model, data, labels, eps, alpha, attack_type):
    """Generate adversarial images from input original images
    
    Args:
        model: An NN model
        data: Original image. Dimension 4: (idx, x, y, channel)
        labels: the labels of the original image. Dimension 1: (idx)
        eps: epsilons in perturbation

    Returns:
        Adversarial image. Dimension 4: (idx, x, y, channel). 
        If prediction is incorrect without any perturbation, ori image would be returned.
    """
    attacker = Attacker(model, data, labels) #
    saving, accuracy, normal_accuracy = attacker.attack(eps, alpha, attack_type, saving_mode = True)
    saving = np.array(list(map(To_noraml_RGB_image, saving)))
    
    # check perturbation min and max
    perturbation = saving - data
    print("Check perturbation range | min:{} | max: {}".format(perturbation.min(), perturbation.max()))
    
    return saving, accuracy, normal_accuracy

def save_adv_image(images, root_path, categorys, images_name):
    """Save all the adversarial image in correspond path 
    
    Args:
        images: Images intended to save
        root_path: A folder intended in save those images
        categorys: the categorys of every image
        images_name: The name of every image

    """
    # create dir if not exist
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    # if exists, delete it and create new
    else:
        shutil.rmtree(root_path)           
        os.makedirs(root_path)
    # create inner folders
    for cate in list(set(categorys)):
        os.mkdir(os.path.join(root_path, cate))
    # save image into corresponding folder
    for i in range(len(images)):
        path = os.path.join(root_path, categorys[i], images_name[i])
        cv2.imwrite(path, images[i])
    print("Save successfully!")

def adversarial_read_gen_save(image_path, model, eps, alpha, attack_type, adv_output):
    """A pipeline to read image and save adversarial image
    
    Args:
        image_path: The folder of original images
        model: An NN model
        eps: epsilons in perturbation
        adv_output: Output folder

    """
    print("=====Read images=====")
    data, labels, categorys, images_name = read_eval_image(image_path)
    print(data.shape)
    print("=====Make adv images=====")
    saving, accuracy, normal_accuracy = generate_saving_image(model, data, labels, eps, alpha, attack_type)
    print("=====Start to save images=====")
    save_adv_image(saving, adv_output, categorys, images_name)
    return accuracy, normal_accuracy

class Adverdataset(Dataset):
    def __init__(self, data, label, transforms):
        self.data = data
        self.label = torch.from_numpy(label).long()
        self.transforms = transforms

    def __getitem__(self, idx):
        img = self.transforms(self.data[idx])
        label = self.label[idx]
        return img, label
    
    def __len__(self):
        return len(self.data)

class Attacker:
    def __init__(self, model, data, label):
        self.data_len = len(data)
        self.model = model
        self.model.to(device)
        self.model.eval()
        self.mean = [0.4914, 0.4822, 0.4465]
        self.std = [0.2023, 0.1994, 0.2010]
        self.normalize = transforms.Normalize(self.mean, self.std, inplace=False)
        transform = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.ToTensor(),
                        self.normalize
                    ])
        self.dataset = Adverdataset(data, label, transform)
        self.loader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size = 1,
                shuffle = False)
        print("dataloader", len(self.loader))

    # FGSM
    def fgsm_attack(self, image, epsilon, data_grad):
        
        # find direction of gradient
        sign_data_grad = data_grad.sign()
        
        # add noise "epilon * direction" to the ori image
        perturbed_image = image + epsilon * sign_data_grad
        
        return perturbed_image
    
    # PGD
    def pgd_attack(self, image, ori_image, eps, alpha, data_grad) :
        
        adv_image = image + alpha * data_grad.sign()
        eta = torch.clamp(adv_image - ori_image.data, min=-eps, max=eps)
        image = ori_image + eta

        return image
    
    def attack(self, epsilon, alpha, attack_type = "FGSM", saving_mode = False):
        adv_all = []
        # save some to show
        adv_examples = []
        wrong, fail, success = 0, 0, 0
        for now, (data, target) in enumerate(self.loader):
            print(str(now) + "|" +str(len(self.loader)), end="\r")
            data, target = data.to(device), target.to(device)
            data_raw = data
            
            # initial prediction
            output = self.model(data)
            init_pred = output.max(1, keepdim=True)[1]

            # DO NOT ATTACK if incorrectly-classified
            if init_pred.item() != target.item():
                wrong += 1
                if saving_mode:
                    data_raw = data_raw * torch.tensor(self.std, device = device).view(3, 1, 1) + torch.tensor(self.mean, device = device).view(3, 1, 1)
                    data_raw = data_raw.squeeze().detach().cpu().numpy()
                    adv_all.append(data_raw)
                continue
                
            # ATTACK if correctly-classified
            ############ ATTACK GENERATION ##############
            if attack_type == "FGSM":
                data.requires_grad = True
                output = self.model(data)
                loss = F.nll_loss(output, target)
                self.model.zero_grad()
                loss.backward()
                data_grad = data.grad.data
                
                perturbed_data = self.fgsm_attack(data, epsilon, data_grad)
            
            elif attack_type == "PGD":
                for i in range(40):
                    data.requires_grad = True
                    output = self.model(data)
                    loss = F.nll_loss(output, target)
                    self.model.zero_grad()
                    loss.backward()
                    data_grad = data.grad.data
                    
                    data = self.pgd_attack(data, data_raw, epsilon, alpha, data_grad).detach_()
                perturbed_data = data
            ############ ATTACK GENERATION ##############

            # prediction of adversarial image        
            output = self.model(perturbed_data)
            final_pred = output.max(1, keepdim=True)[1]
            
            # if still correctly-predicting, attack failed
            if final_pred.item() == target.item():
                fail += 1
            
            # incorrectly-predicting, attack successfully
            else:
                success += 1
                
                # save some success adversarial example
                if len(adv_examples) < 5:
                    adv_ex = perturbed_data * torch.tensor(self.std, device = device).view(3, 1, 1) + torch.tensor(self.mean, device = device).view(3, 1, 1)
                    adv_ex = adv_ex.squeeze().detach().cpu().numpy() 
                    data_raw = data_raw * torch.tensor(self.std, device = device).view(3, 1, 1) + torch.tensor(self.mean, device = device).view(3, 1, 1)
                    data_raw = data_raw.squeeze().detach().cpu().numpy()
                    adv_examples.append( (init_pred.item(), final_pred.item(), data_raw , adv_ex) )  
            
            # in the image saving mode, save all the adversarial images
            if saving_mode:
                adv_ex = perturbed_data * torch.tensor(self.std, device = device).view(3, 1, 1) + torch.tensor(self.mean, device = device).view(3, 1, 1)
                adv_ex = adv_ex.squeeze().detach().cpu().numpy() 
                adv_all.append(adv_ex)
        
        # calculate final accuracy 
        final_acc = (fail / (wrong + success + fail))
        print("Epsilon: {}\tTest Accuracy = {} / {} = {}\n".format(epsilon, fail, len(self.loader), final_acc))
        print("Wrong: {} \tFail: {}\tSuccess: {}".format(wrong, fail, success))
        
        if saving_mode:
            noraml_acc = 1 - (wrong / (wrong + success + fail))
            return adv_all, final_acc, noraml_acc
        else:
            return adv_examples, final_acc
