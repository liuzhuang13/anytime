from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import time
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb, time

def json_save(filename, json_obj):
    import json
    with open(filename, 'w') as f:
        json.dump(json_obj, f, indent=4)

def json_read(filename):
    import json
    with open(filename, 'r') as f:
        data = json.load(f)

    return data


class FullModel(nn.Module):
  def __init__(self, model, loss):
    super(FullModel, self).__init__()
    self.model = model
    self.loss = loss

  def forward(self, inputs, labels):
    outputs = self.model(inputs)
    loss = self.loss(outputs, labels)
    return torch.unsqueeze(loss,0), outputs


class FullEEModel(nn.Module):
  def __init__(self, model, loss, config=None):
    super(FullEEModel, self).__init__()
    self.model = model
    self.loss = loss
    self.cfg = config

  def forward(self, inputs, labels):
    outputs = self.model(inputs)
    losses = []
    for i, output in enumerate(outputs):
        losses.append(self.loss(outputs[i], labels))

    return losses, outputs

def get_world_size():
    if not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size()

def get_rank():
    if not torch.distributed.is_initialized():
        return 0
    return torch.distributed.get_rank()

class AverageMeter(object):
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

def create_logger(cfg, cfg_name, phase='train'):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    dataset = cfg.DATASET.DATASET
    model = cfg.MODEL.NAME
    cfg_name = os.path.basename(cfg_name).split('.')[0]
    final_output_dir = root_output_dir
    os.makedirs(final_output_dir, exist_ok=True)

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)
    tensorboard_log_dir = final_output_dir
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)


def get_confusion_matrix_gpu(label, pred, size, num_class, ignore=-1, device=0):
    output = pred.transpose(1,3).transpose(1,2)
    seg_pred = torch.max(output, dim=3)[1]
    seg_gt = label

    ignore_index = seg_gt != ignore

    seg_gt = seg_gt[ignore_index]

    seg_pred = seg_pred[ignore_index]
    if seg_gt.get_device() == -1:
        seg_gt = seg_gt.to(0)

    index = ((seg_gt * num_class).long()+ seg_pred)
    label_count = torch.bincount(index)
    confusion_matrix = np.zeros((num_class, num_class))

    for i_label in range(num_class):
        for i_pred in range(num_class):
            cur_index = i_label * num_class + i_pred
            if cur_index < len(label_count):
                confusion_matrix[i_label,
                                 i_pred] = label_count[cur_index]
    return confusion_matrix

def adjust_learning_rate(optimizer, base_lr, max_iters, 
        cur_iters, power=0.9):
    lr = base_lr*((1-float(cur_iters)/max_iters)**(power))
    for i, param in enumerate(optimizer.param_groups):
        optimizer.param_groups[i]['lr'] = lr
    return lr