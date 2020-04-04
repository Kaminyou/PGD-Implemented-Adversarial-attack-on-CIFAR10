# PGD-Implemented-Adversarial-attack-on-CIFAR10
An example code of implement of PGD and FGSM algorithm for adversarial attack

## Pretrained model
The pretrained models is from [here](https://github.com/huyvnphan/PyTorch-CIFAR10) <br />
Please download the pretrained models first and put them in the /cifar10_models/state_dicts as instruction in above link.<br />

## Prepare normal examples
Please prepare your cifar-10 normal example and put all the classes in different folders. That is:<br />
<a>| imgs/</a><br />
<a>  | - frog</a><br />
<a>    | -- frog1.png  frog2.png  ......</a><br />
<a>        ...</a><br />
<a>  | - automobile</a><br />
<a>    ...</a><br />
## Generate adversarial example
<br />
    $python3 main.py -I input_normal_examples_path -M model -T mode -O adversarial_examples_folder_name
<br />
model: vgg16_bn, resnet50, mobilenet_v2, densenet161<br />
mode: PGD, FGSM<br />
<br />


## Investigate transferability
<br />
    $python3 transferability.py -I input_normal_examples_path -O 1or0
<br />
O: if generate confusion table or not
