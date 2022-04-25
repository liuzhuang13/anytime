from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from yacs.config import CfgNode as CN


_C = CN(new_allowed=True)

_C.OUTPUT_DIR = ''
_C.LOG_DIR = ''
_C.GPUS = (0,)
_C.WORKERS = 4
_C.PRINT_FREQ = 20
_C.AUTO_RESUME = False
_C.PIN_MEMORY = True
_C.RANK = 0

_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

_C.MODEL = CN(new_allowed=True)
_C.MODEL.NAME = 'seg_hrnet'
_C.MODEL.PRETRAINED = ''
_C.MODEL.LOAD_STAGE = 0
_C.MODEL.EXTRA = CN(new_allowed=True)

_C.LOSS = CN()
_C.LOSS.USE_OHEM = False
_C.LOSS.OHEMTHRES = 0.9
_C.LOSS.OHEMKEEP = 100000
_C.LOSS.CLASS_BALANCE = True

_C.DATASET = CN()
_C.DATASET.ROOT = ''
_C.DATASET.DATASET = 'cityscapes'
_C.DATASET.NUM_CLASSES = 19
_C.DATASET.TRAIN_SET = 'list/cityscapes/train.lst'
_C.DATASET.EXTRA_TRAIN_SET = ''
_C.DATASET.TEST_SET = 'list/cityscapes/val.lst'

_C.TRAIN = CN(new_allowed=True)

_C.TRAIN.IMAGE_SIZE = [1024, 512]
_C.TRAIN.BASE_SIZE = 2048
_C.TRAIN.DOWNSAMPLERATE = 1
_C.TRAIN.FLIP = True
_C.TRAIN.MULTI_SCALE = True
_C.TRAIN.SCALE_FACTOR = 16

_C.TRAIN.LR_FACTOR = 0.1
_C.TRAIN.LR_STEP = [90, 110]
_C.TRAIN.LR = 0.01
_C.TRAIN.EXTRA_LR = 0.001

_C.TRAIN.OPTIMIZER = 'sgd'
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WD = 0.0001
_C.TRAIN.NESTEROV = False
_C.TRAIN.IGNORE_LABEL = -1

_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 484
_C.TRAIN.EXTRA_EPOCH = 0

_C.TRAIN.RESUME = False

_C.TRAIN.BATCH_SIZE_PER_GPU = 32
_C.TRAIN.SHUFFLE = True
_C.TRAIN.NUM_SAMPLES = 0

_C.TEST = CN(new_allowed=True)

_C.TEST.IMAGE_SIZE = [2048, 1024]
_C.TEST.BASE_SIZE = 2048

_C.TEST.BATCH_SIZE_PER_GPU = 32
_C.TEST.NUM_SAMPLES = 0

_C.TEST.MODEL_FILE = ''
_C.TEST.FLIP_TEST = False
_C.TEST.MULTI_SCALE = False
_C.TEST.CENTER_CROP_TEST = False
_C.TEST.SCALE_LIST = [1]

_C.DEBUG = CN()
_C.DEBUG.DEBUG = False
_C.DEBUG.SAVE_BATCH_IMAGES_GT = False
_C.DEBUG.SAVE_BATCH_IMAGES_PRED = False
_C.DEBUG.SAVE_HEATMAPS_GT = False
_C.DEBUG.SAVE_HEATMAPS_PRED = False

_C.EXIT = CN(new_allowed=True)

_C.EXIT.TYPE = 'original'
_C.EXIT.FINAL_CONV_KERNEL = 1
_C.EXIT.COMP_RATE = 1.0
_C.EXIT.SMOOTH = False
_C.EXIT.SMOOTH_KS = 3
_C.EXIT.LAST_SAME = False
_C.EXIT.FIX_INTER_CHANNEL = False
_C.EXIT.INTER_CHANNEL = 64

_C.MASK = CN(new_allowed=True)
_C.MASK.ENTROPY_THRE = 0.0

_C.PYRAMID_TEST = CN(new_allowed=True)
_C.PYRAMID_TEST.USE = False
_C.PYRAMID_TEST.SIZE = 512



def update_config(cfg, args):
    cfg.defrost()
    
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    cfg.freeze()


if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)

