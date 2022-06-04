import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import os
import argparse
from torch.optim import Adam



class Base(object):

    def __init__(self, args):
        super(Base, self).__init__()
        self.args = None
        self.model = None
        self.loss_fn = None
        self.lr = None
        self.optimizer = None
        self.lr_decay = None
        self.output = None
        self.loss_val = None
        self.epoch = self.args.epoch
        self.batch_size = self.args.batch_size
        self.pre_trained = self.args.pre_trained
        self.root_dir = self.args.root_dir
        self.crop_size = self.args.crop_size
        self.exp_dir = self.args.exp_dir
        self.save_model_name = self.args.save_model_name
        self.best_psnr = 0.
        self.best_epoch = 0
        self.step = self.args.step