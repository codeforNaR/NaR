import torch.nn as nn
import torch

import os
import shutil
import models.cifar as models
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import numpy as np
import random

class MutualModel(nn.Module):
    def __init__(self, models):
        super(MutualModel, self).__init__()
        self.models = nn.ModuleList(models)
        self.size = len(models)
    
    def forward(self, x):
        return [model(x) for model in self.models]
        
def build_model(arch, num_classes, depth, args):
    if arch.startswith('resnext'):
        model = models.__dict__[arch](
                    cardinality=args.cardinality,
                    num_classes=num_classes,
                    depth=depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                )
    elif arch.startswith('densenet'):
        model = models.__dict__[arch](
                    num_classes=num_classes,
                    depth=depth,
                    growthRate=args.growthRate,
                    compressionRate=args.compressionRate,
                    dropRate=args.drop,
                )
    elif arch.startswith('wrn'):
        model = models.__dict__[arch](
                    num_classes=num_classes,
                    depth=depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                )
    elif arch.endswith('resnet'):
        model = models.__dict__[arch](
                    num_classes=num_classes,
                    depth=depth)

    elif arch.startswith('plain'):
        model = models.__dict__[arch](num_classes=num_classes,
                        depth=depth)
    else:
        model = models.__dict__[arch](num_classes=num_classes)
    return model 


def build_dataloaders(args):
    # Data
    print('==> Preparing dataset %s' % args.dataset)
    if args.no_aug:
        transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    else:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    def _init_fn(worker_id):
        np.random.seed(args.manualSeed)

    if args.dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        #mydataset = myds.CIFAR10
        num_classes = 10
        args.K = 2
    else:
        dataloader = datasets.CIFAR100
        #mydataset = myds.CIFAR100
        num_classes = 100

    trainset = dataloader(root='./data', train=True, download=True, transform=transform_train)
    #trainset2 = mydataset(root='./data', train=True, transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers,
        worker_init_fn=_init_fn)

    testset = dataloader(root='./data', train=False, download=False, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        worker_init_fn=_init_fn)

    return trainloader, testloader


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def seed_torch(seed=201905):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
