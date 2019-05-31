'''
Training script for CIFAR-10/100
'''
from __future__ import print_function

import argparse
import os
import shutil
import time
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models.cifar as models
from utils.meter import MultAverageMeter


from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig

from utils.losses import LabelSmoothing, NaRCriterion
from common_args import arg_parse
from trainutils import *

args = arg_parse()

state = {k: v for k, v in args._get_kwargs()}

# Validate dataset
assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'

#set checkpoints
args.checkpoint = os.path.join('./checkpoints/nar/',args.dataset,args.arch+'-'+str(args.depth)+'-'+args.arch2+'-'+args.memo+'-'+ \
                    time.strftime('%Y-%m-%d-%H%M%S',time.localtime(time.time())))
# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
seed_torch(args.manualSeed)

best_acc = [0,0]  # best test accuracy


def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    if args.dataset == 'cifar10':
        num_classes = 10
        epislon = 0.3
    else:
        num_classes = 100
        epislon = 0.5
   
    trainloader, testloader = build_dataloaders(args)
   
    # Model
    print("==> creating target network '{}'".format(args.arch))
    net1 = build_model(args.arch, num_classes, args.depth, args)
    print("==> creating auxiliary network '{}'".format(args.arch2))
    net2 = build_model(args.arch2, num_classes, args.depth2, args)
    model = MutualModel([net1, net2])

    model = model.cuda()
    
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    criterion = NaRCriterion(args.alpha, args.lam, nb_classes=num_classes, epislon=epislon)
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay, nesterov=True)
    
    # Resume
    title = args.dataset +  '-' + args.arch
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'M1 Train Acc.','M2 Train Acc.', 'M1 Valid Acc.', 'M2 Valid Acc.'])


    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test(testloader, model, criterion, start_epoch, use_cuda)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return

    # Train and val
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_loss, train_acc = train(trainloader, model, criterion, optimizer, epoch, use_cuda)
        test_loss, test_acc = test(testloader, model, criterion, epoch, use_cuda)

        # append logger file
        logger.append([state['lr'], train_loss, test_loss, train_acc[0], train_acc[1], test_acc[0], test_acc[1]])

        # save model
        is_best = test_acc[0] > best_acc[0] or test_acc[1] > best_acc[1]
        best_acc = [max(test_acc[0], best_acc[0]), max(test_acc[1], best_acc[1])]
        if args.save_intermediate and (epoch+1)% args.ckpt_interval ==0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict':model.state_dict(),
            }, False, checkpoint=args.checkpoint,
                filename='ckpt_%i.pt'%(epoch+1))
        # routine save
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint=args.checkpoint)

    logger.close()
    #logger.plot()
   
    print('Best acc:')
    print(best_acc)

def train(trainloader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = MultAverageMeter()
    top5 = MultAverageMeter()
    end = time.time()

    use_warm_start = True if epoch <20 else False
    bar = Bar('Processing', max=len(trainloader))

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
    
        # compute output
        outputs = model(inputs)
       
        loss = criterion(outputs, targets, warm=use_warm_start)
        losses.update(loss.item(), inputs.size(0))
        # measure accuracy and record loss
        prec1, prec5 = [], []
        for i in range(len(outputs)):
            prec1_, prec5_ = accuracy(outputs[i], targets, topk=(1, 5))
            prec1.append(prec1_.item())
            prec5.append(prec5_.item())
        top1.update(prec1, inputs.size(0))
        top5.update(prec5, inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f}, {top1_2: .4f} | top5: {top5: .4f}, {top5_2: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg[0],
                    top1_2=top1.avg[1],
                    top5=top5.avg[0],
                    top5_2=top5.avg[1]
                    )
        bar.next()
    bar.finish()
   
    
    return (losses.avg, top1.avg)

def test(testloader, model, criterion, epoch, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = MultAverageMeter()
    top5 = MultAverageMeter()

    # switch to evaluate mode
    model.eval()
    use_warm_start = True if epoch <20 else False

    end = time.time()
    bar = Bar('Processing', max=len(testloader))
    total_pred = []
    total_target = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            # measure data loading time
            data_time.update(time.time() - end)

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            # compute output
            outputs = model(inputs)

            total_pred.append([o.detach() for o in outputs])
            total_target.append(targets)

            loss = criterion(outputs, targets, warm=use_warm_start)

            losses.update(loss.item(), inputs.size(0))
            # measure accuracy and record loss
            prec1, prec5 = [], []
            for i in range(len(outputs)):
                prec1_, prec5_ = accuracy(outputs[i], targets, topk=(1, 5))
                prec1.append(prec1_.item())
                prec5.append(prec5_.item())
            top1.update(prec1, inputs.size(0))
            top5.update(prec5, inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f}, {top1_2: .4f} | top5: {top5: .4f}, {top5_2: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(testloader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg,
                        top1=top1.avg[0],
                        top1_2=top1.avg[1],
                        top5=top5.avg[0],
                        top5_2=top5.avg[1]
                        )
            bar.next()
    bar.finish()
    
    return (losses.avg, top1.avg)

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

if __name__ == '__main__':
    main()
