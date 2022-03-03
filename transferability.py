from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torchvision.transforms as transforms

from cifar10_models import *
from utils import *


class TransferAttacker:
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
                batch_size = 20,
                shuffle = False)
        
    def evaluation(self):
        accuracy = 0
        for now, (data, target) in enumerate(self.loader):
            print(str(now) + "|" +str(len(self.loader)), end="\r")
            data, target = data.to(device), target.to(device)
            data_raw = data
            
            # initial prediction
            output = self.model(data)
            init_pred = output.max(1, keepdim=True)[1].squeeze()
            accuracy += torch.sum(init_pred == target).cpu().numpy()
        print("Accuracy: {} / {}".format(accuracy, self.data_len))
        accuracy = accuracy/self.data_len
        return accuracy

def transfer_attack(image_path, model_name, pretrained = True):
    """
    Evaluation model with given data
    image_path: path of the data(image)
    model_name: vgg16_bn, resnet50, mobilenet_v2, densenet161
    
    """
    model_test = select_model(model_name, pretrained)
    data, labels, categorys, images_name = read_eval_image(image_path)
    evaluation = TransferAttacker(model_test, data, labels)
    accuracy = evaluation.evaluation()
    return accuracy

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
