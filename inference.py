import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import time, math, json, os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from args import get_train_args
import numpy as np
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import csv


def main():
    opts = get_train_args()
    print ("load data ...")
    train_data = datasets.ImageFolder(root="data/train",
                           transform=transforms.Compose([
                               transforms.Resize((256,256)),       # 한 축을 128로 조절하고
                               #transforms.CenterCrop(256),  # square를 한 후,
                               transforms.ToTensor(),       # Tensor로 바꾸고 (0~1로 자동으로 normalize)
                               transforms.Normalize((0.5, 0.5, 0.5),  # -1 ~ 1 사이로 normalize
                                                    (0.5, 0.5, 0.5)), # (c - m)/s 니까...
                           ]))

    test_data = datasets.ImageFolder(root="data/test",
                           transform=transforms.Compose([
                               transforms.Resize((256,256)),       # 한 축을 128로 조절하고
                               #transforms.CenterCrop(256),  # square를 한 후,
                               transforms.ToTensor(),       # Tensor로 바꾸고 (0~1로 자동으로 normalize)
                               transforms.Normalize((0.5, 0.5, 0.5),  # -1 ~ 1 사이로 normalize
                                                    (0.5, 0.5, 0.5)), # (c - m)/s 니까...
                           ]))
    test_loader = DataLoader(test_data,
                             batch_size=opts.batch_size,
                             shuffle=False,
                             num_workers=opts.num_processes)

    classes = train_data.classes
    print (classes)

    print ("load model ...")
    if opts.model == 'resnet':
        model = models.resnet50(progress=True)
        model.load_state_dict(torch.load('resnet_model.pt'))
    elif opts.model == 'vggnet':
        model = models.vgg13_bn(progress=True)
        model.load_state_dict(torch.load('vggnet_model.pt'))
    elif opts.model == 'googlenet':
        model = models.googlenet(progress=True)
        model.load_state_dict(torch.load('googlenet_model.pt'))
    elif opts.model == 'densenet':
        model = models.densenet121(progress=True)
        model.load_state_dict(torch.load('densenet_model.pt'))
    else:
        model = models.resnext50_32x4d(progress=True)
        model.load_state_dict(torch.load('resnext_model.pt'))
    print (opts.model)
    model.cuda()

    print ("start inference")

    idx = 0
    with torch.no_grad():
        with open(opts.model +'_result.txt', 'a') as f:
            for i, (inputs,labels) in enumerate(test_loader):
                inputs = inputs.cuda()
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                for j, meta in enumerate(predicted):
                    predicted_class = classes[meta]
                    plant_class = predicted_class.split('_')[0]
                    disease_class = predicted_class.split('_')[1]
                    f.write(str(test_loader.dataset.samples[idx][0].split('/')[-1].split('.')[0]) + '\t' + plant_class + '\t' + disease_class + '\n')
                    idx += 1


if __name__ == '__main__':
    main()