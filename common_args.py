import argparse

import models.cifar as models
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

mode_pools = ['soft','hard','smoothing','penalty','simple','disturb']

def arg_parse():

    parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
    parser.add_argument('--memo', default='test',type=str)
    # Datasets
    parser.add_argument('-d', '--dataset', default='cifar100', type=str)
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    # Optimization options
    parser.add_argument('--epochs', default=300, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                        help='train batchsize')
    parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                        help='test batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--drop', '--dropout', default=0, type=float,
                        metavar='Dropout', help='Dropout ratio')
    parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                            help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    # Checkpoints
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--resume2', default='', type=str, metavar='PATH')
    parser.add_argument('--snap-interval', type=int, default=20)
    # Architecture
    parser.add_argument('--arch2', default='resnet32', type=str)
    parser.add_argument('--arch', '-a', default='resnet20',type=str,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: resnet18)')
    parser.add_argument('--depth', type=int, default=29, help='Model depth.')
    parser.add_argument('--depth2', type=int, default=29, help='Model depth.')
    parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
    parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
    parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
    parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
    # Miscs

    parser.add_argument('--silence', action='store_true')
    parser.add_argument('-K', default=5, type=int, help='top k accuracy')
    parser.add_argument('--warmup', default=20, type=int, help='top k accuracy')
    parser.add_argument('--use-warm', action='store_true')

    parser.add_argument('-T', type=float, default=1.0, help='tempature')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    #Device options
    parser.add_argument('--no-aug', action='store_true', help='no augmentation')
    parser.add_argument('--gpu-id', default='0', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--reg-mode', default='simple', 
                    choices=mode_pools,
                     help='type of regularization')
    parser.add_argument('--save-intermediate', action='store_true')
    parser.add_argument('--ckpt-interval', default=20, type=int)
    parser.add_argument('--alpha', default=0.1, type=float)
    parser.add_argument('--lam', default=0.5, type=float)
    args = parser.parse_args()

    return args
