from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import functools

import numpy as np

import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F

from utils.utils import AverageMeter
import pdb, time
from .conv_mask import conv_mask_uniform
from functools import partial

from utils.utils import get_rank

BatchNorm2d = nn.BatchNorm2d
BN_MOMENTUM = 0.01
logger = logging.getLogger(__name__)

def conv3x3(in_planes, out_planes, stride=1):
    return used_conv(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = used_conv(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = used_conv(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = used_conv(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = BatchNorm2d(planes * self.expansion,
                               momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
           self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                used_conv(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(num_channels[branch_index] * block.expansion,
                            momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        used_conv(num_inchannels[j],
                                  num_inchannels[i],
                                  1,
                                  1,
                                  0,
                                  bias=False),
                        BatchNorm2d(num_inchannels[i], momentum=BN_MOMENTUM)))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                used_conv(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                BatchNorm2d(num_outchannels_conv3x3, 
                                            momentum=BN_MOMENTUM)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                used_conv(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                BatchNorm2d(num_outchannels_conv3x3,
                                            momentum=BN_MOMENTUM),
                                nn.ReLU(inplace=True)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=[height_output, width_output],
                        mode='bilinear')
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}



class HighResolutionNet(nn.Module):
    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        used_conv(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        BatchNorm2d(
                            num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        used_conv(
                            inchannels, outchannels, 3, 2, 1, bias=False),
                        BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                used_conv(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(
                HighResolutionModule(num_branches,
                                      block,
                                      num_blocks,
                                      num_inchannels,
                                      num_channels,
                                      fuse_method,
                                      reset_multi_scale_output)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def __init__(self, config, **kwargs):


        super(HighResolutionNet, self).__init__()
        extra = config.MODEL.EXTRA
        self.extra = extra
        self.mask_cfg = config.MASK

        global mask_conv, mask_conv_no_interpolate
        mask_conv = partial(conv_mask_uniform, p=self.mask_cfg.P, interpolate=self.mask_cfg.INTERPOLATION)
        mask_conv_no_interpolate = partial(conv_mask_uniform, p=self.mask_cfg.P, interpolate='none')
        global used_conv
        used_conv = nn.Conv2d

        self.num_exits = len(extra.EE_WEIGHTS)
        self.num_classes = config.DATASET.NUM_CLASSES
        if 'profiling_cpu' in kwargs or 'profiling_gpu' in kwargs:
            self.profiling_meters = [AverageMeter() for i in range(self.num_exits)]
            self.profiling_gpu = 'profiling_gpu' in kwargs
            self.profiling_cpu = 'profiling_cpu' in kwargs
            self.forward_count = 0
        else:
            self.profiling_gpu, self.profiling_cpu = False, False

        self.conv1 = used_conv(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = used_conv(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        
        self.stage1_cfg = extra['STAGE1']
        num_channels = self.stage1_cfg['NUM_CHANNELS'][0]
        block = blocks_dict[self.stage1_cfg['BLOCK']]
        num_blocks = self.stage1_cfg['NUM_BLOCKS'][0]
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)
        stage1_out_channel = block.expansion*num_channels
        self.exit1 = self.get_exit_layer(stage1_out_channel, config, exit_number=1)

        if self.mask_cfg.USE:
            used_conv = mask_conv
        else:
            used_conv = nn.Conv2d

        self.stage2_cfg = extra['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer(
            [stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)
        self.exit2 = self.get_exit_layer(np.int(np.sum(pre_stage_channels)), config, exit_number=2)

        if self.mask_cfg.USE:
            used_conv = mask_conv
        else:
            used_conv = nn.Conv2d


        self.stage3_cfg = extra['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)
        self.exit3 = self.get_exit_layer(np.int(np.sum(pre_stage_channels)), config, exit_number=3)

        self.stage4_cfg = extra['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=True)
        
        last_inp_channels = np.int(np.sum(pre_stage_channels))
        self.last_layer = self.get_exit_layer(last_inp_channels, config, last=True)

        print(sum(p.numel() for p in self.parameters() if p.requires_grad))
        print(sum(p.numel() for p in self.parameters()))



    def profile(self, out, index):
        if not (self.profiling_cpu or self.profiling_gpu):
            return
        self.forward_count += 1
        print(self.forward_count)
        start_count = 25 * 4
        if self.forward_count < start_count:
            return

        if self.profiling_cpu:
            self.profiling_meters[index].update(time.time() - self.start)
        elif self.profiling_gpu:
            tmp_out = out.cpu()
            torch.cuda.synchronize()
            self.profiling_meters[index].update(time.time() - self.start)
        else:
            return
        if index == self.num_exits - 1 and (self.forward_count > start_count + 10):
            times = [self.profiling_meters[i].average() for i in range(self.num_exits)]
            times.append(np.mean(times))
            print('\t'.join(['{:.3f}'.format(x) for x in times]))

    def get_points_from_confs(self, confs, ratio):
        bs, h, w = confs.size(0), confs.size(2), confs.size(3)
        idx = torch.arange(h * w, device=confs.device)
        h_pos = idx // w
        w_pos = idx % w
        point_coords_int = torch.cat((h_pos.unsqueeze(1), w_pos.unsqueeze(1)), dim=1)
        point_coords_int = point_coords_int.unsqueeze(0).repeat(bs, 1, 1)
        num_sampled = point_coords_int.size(1)

        num_certain_points = int(ratio * h * w)
        point_certainties = confs.view(bs, 1, -1)
        values, idx = torch.topk(point_certainties[:, 0, :], k=num_certain_points, dim=1)
        shift = num_sampled * torch.arange(bs, dtype=torch.long, device=confs.device)
        idx += shift[:, None]
        point_coords_selected_int = point_coords_int.view(-1, 2)[idx.view(-1), :].view(
                bs, num_certain_points, 2
            )
        point_coords_selected_frac = torch.cat(( (point_coords_selected_int[:, :, 0:1] + 0.5)/float(h), (point_coords_selected_int[:, :, 1:2] + 0.5)/float(w)), dim=2)
        return point_coords_selected_int, point_coords_selected_frac

    def get_resized_mask_from_logits(self, logits, h, w,criterion):
        if criterion == 'conf_thre':
            resized_logits = F.interpolate(logits, size=(h, w))
            resized_probs = F.softmax(resized_logits, dim=1)
            resized_confs, _ = resized_probs.max(dim=1, keepdim=True)
            mask = (resized_confs <= self.mask_cfg.CONF_THRE).float().view(logits.size(0), h, w)
        elif criterion == 'entropy_thre':
            resized_logits = F.interpolate(logits, size=(h, w))
            resized_probs = F.softmax(resized_logits, dim=1)
            resized_confs =  torch.sum( - resized_probs * torch.log(resized_probs), dim=1, keepdim=True) # 
            mask = (resized_confs >= self.mask_cfg.ENTROPY_THRE).float().view(logits.size(0), h, w)
        return mask

    def generate_grid_priors(self):
        if hasattr(self, 'mask_grid_prior_dict') and len(self.mask_grid_prior_dict) > 0:
            return
        self.mask_grid_prior_dict = {}

        for m in self.modules():
            if isinstance(m, conv_mask_uniform):
                try:
                    h,w = m.out_h, m.out_w
                except:
                    logger.info("First forwarding, collecting output size, quit generating grid priors")
                    break

                if (h,w) in self.mask_grid_prior_dict:
                    continue
                logger.info(f"generating grid priors for size {(h,w)}")
                res = torch.zeros((h, w), device=m.weight.device)
                stride = self.mask_cfg.GRID_STRIDE
                start = (stride - 1) // 2

                for i in range(start, res.size(0), stride):
                    for j in range(start, res.size(1), stride):
                        res[i][j] = 1.

                self.mask_grid_prior_dict[(h, w)] = res

    def set_masks(self, logits):
        self.mask_dict = {}
        for m in self.modules():
            if isinstance(m, conv_mask_uniform):
                try:
                    h,w = m.out_h, m.out_w
                except:
                    logger.info("First forwarding, collecting output size, quit setting masks")
                    break

                if (h,w) in self.mask_dict:
                    m.set_mask(self.mask_dict[(h,w)])
                else:
                    self.mask_dict[(h,w)] = self.get_resized_mask_from_logits(logits, h, w, criterion=self.mask_cfg.CRIT)

                    if self.mask_cfg.GRID_PRIOR:
                        self.mask_dict[(h,w)] = torch.max(self.mask_dict[(h,w)], self.mask_grid_prior_dict[(h, w)])
                    m.set_mask(self.mask_dict[(h,w)])


    def set_part_masks(self, logits, ref_name, masked_modules):
        start = time.time()
        self.part_mask_dicts[ref_name] = {} 
        for module in masked_modules:
            for m in module.modules():
                if isinstance(m, conv_mask_uniform):
                    try:
                        h,w = m.out_h, m.out_w
                    except:
                        logger.info("First forwarding, collecting output size, quit setting masks")
                        break
                    if (h,w) in self.part_mask_dicts[ref_name]:
                        m.set_mask(self.part_mask_dicts[ref_name][(h,w)])
                    else:
                        self.part_mask_dicts[ref_name][(h,w)] = self.get_resized_mask_from_logits(logits, h, w, criterion=self.mask_cfg.CRIT)
                        m.set_mask(self.part_mask_dicts[ref_name][(h,w)])

    def forward(self, x):
        self.part_mask_dicts = {}

        if self.profiling_gpu:
            torch.cuda.synchronize()
        if self.profiling_cpu or self.profiling_gpu:
            self.start = time.time()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x) 
        out1_feat = self.get_exit_input([x], detach=self.extra.EARLY_DETACH)
        out1 = self.exit1(out1_feat) # logits of exit 1
        out_size = (out1.size(-2), out1.size(-1))

        # Set mask for all conv_mask modules between exit 1 and exit 2
        if self.mask_cfg.USE:
            self.set_part_masks(out1, 'out1', [self.transition1, self.stage2, self.exit2])
        if hasattr(self, "stop1"):
            return out1

        x_list = [] 
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)
        out2_feat = self.get_exit_input(y_list, detach=self.extra.EARLY_DETACH)
        out2 = self.exit2(out2_feat)

        if self.mask_cfg.USE:
            # Compute logits, aggregate results from the previous exit
            if self.mask_cfg.AGGR == 'copy' and len(self.part_mask_dicts['out1']) > 0:
                result_mask = self.part_mask_dicts['out1'][out_size][:, None, :, :]
                out2 = out1 * (1-result_mask) + out2 * result_mask 
            # Set mask for all conv_mask modules between exit 2 and exit 3
            self.set_part_masks(out2, 'out2', [self.transition2, self.stage3, self.exit3])
        if hasattr(self, "stop2"):
            return out2

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)
        out3_feat = self.get_exit_input(y_list, detach=self.extra.EARLY_DETACH)
        out3 = self.exit3(out3_feat)

        if self.mask_cfg.USE:
            # Compute logits, aggregate results from the previous exit
            if self.mask_cfg.AGGR == 'copy' and len(self.part_mask_dicts['out2']) > 0:
                result_mask = self.part_mask_dicts['out2'][out_size][:, None, :, :]
                out3 = out2 * (1-result_mask) + out3 * result_mask
             # Set mask for all conv_mask module between exit 3 and exit 4
            self.set_part_masks(out3, 'out3', [self.transition3, self.stage4, self.last_layer])
        if hasattr(self, "stop3"):
            return out3

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])

        y_list = self.stage4(x_list)
        out4_feat = self.get_exit_input(y_list, detach=False)
        out4 = self.last_layer(out4_feat)

        if self.mask_cfg.USE:
            if self.mask_cfg.AGGR == 'copy' and len(self.part_mask_dicts['out3']) > 0:
                result_mask = self.part_mask_dicts['out3'][out_size][:, None, :, :]
                out4 = out3 * (1-result_mask) + out4 * result_mask

        self.profile(out4, 3) 
        if hasattr(self, "stop4"):
            return out4

        outs = [out1, out2, out3, out4]

        return outs


    def get_exit_layer(self, num_channels, config, last=False, exit_number=0):
        print(f'EXIT num_channels:{num_channels}')
        extra = config.MODEL.EXTRA
        layer_type = config.EXIT.TYPE if (not last) else 'original'

        inter_channel = int(num_channels)

        if layer_type == 'flex':
            assert exit_number in [1,2,3]
            type_map = {1: 'downup_pool_1x1_inter_triple', 2: 'downup_pool_1x1_inter_double', 3: 'downup_pool_1x1_inter'}
            layer_type = type_map[exit_number]
            inter_channel = config.EXIT.INTER_CHANNEL

        if self.mask_cfg.USE:
            exit_conv = used_conv
        else:
            exit_conv = nn.Conv2d

        norm_layer = BatchNorm2d(num_channels, momentum=BN_MOMENTUM)

        if layer_type == 'original':
            exit_layer = [
                exit_conv(
                    in_channels=num_channels,
                    out_channels=num_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True),
                
                norm_layer,
                nn.ReLU(inplace=True), 
                exit_conv( 
                    in_channels=num_channels,
                    out_channels=config.DATASET.NUM_CLASSES,
                    kernel_size=config.EXIT.FINAL_CONV_KERNEL,
                    stride=1,
                    padding=1 if config.EXIT.FINAL_CONV_KERNEL == 3 else 0,
                    bias=True, 
                    )
            ]

        elif layer_type == 'downup_pool_1x1_inter':
            exit_layer = [
                nn.AvgPool2d(2, 2),
                exit_conv(
                    in_channels=num_channels,
                    out_channels=inter_channel,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True),
                BatchNorm2d(inter_channel, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True), 
                nn.Upsample(scale_factor=2, mode='bilinear'),
                exit_conv(
                    in_channels=inter_channel,
                    out_channels=config.DATASET.NUM_CLASSES,
                    kernel_size=config.EXIT.FINAL_CONV_KERNEL,
                    stride=1,
                    padding=1 if config.EXIT.FINAL_CONV_KERNEL == 3 else 0,
                    bias=True, 
                    )
            ]



        elif layer_type == 'downup_pool_1x1_inter_double':
            exit_layer = [
                nn.AvgPool2d(2, 2),
                exit_conv(
                    in_channels=num_channels,
                    out_channels=inter_channel,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True),
                BatchNorm2d(inter_channel, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True), 

                nn.AvgPool2d(2, 2),
                exit_conv(
                    in_channels=inter_channel,
                    out_channels=inter_channel,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True
                    ),
                BatchNorm2d(inter_channel, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),

                nn.Upsample(scale_factor=2, mode='bilinear'),
                exit_conv(
                    in_channels=inter_channel,
                    out_channels=inter_channel,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True
                    ),
                BatchNorm2d(inter_channel, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),

                nn.Upsample(scale_factor=2, mode='bilinear'),
                exit_conv(
                    in_channels=inter_channel,
                    out_channels=config.DATASET.NUM_CLASSES,
                    kernel_size=config.EXIT.FINAL_CONV_KERNEL,
                    stride=1,
                    padding=1 if config.EXIT.FINAL_CONV_KERNEL == 3 else 0,
                    bias=True, 
                    )
            ]


        elif layer_type == 'downup_pool_1x1_inter_triple':
            exit_layer = [
                nn.AvgPool2d(2, 2),
                exit_conv(
                    in_channels=num_channels,
                    out_channels=inter_channel,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True),
                BatchNorm2d(inter_channel, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True), 

                nn.AvgPool2d(2, 2),
                exit_conv(
                    in_channels=inter_channel,
                    out_channels=inter_channel,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True
                    ),
                BatchNorm2d(inter_channel, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),

                nn.AvgPool2d(2, 2),
                exit_conv(
                    in_channels=inter_channel,
                    out_channels=inter_channel,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True
                    ),
                BatchNorm2d(inter_channel, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),

                nn.Upsample(scale_factor=2, mode='bilinear'),
                exit_conv(
                    in_channels=inter_channel,
                    out_channels=inter_channel,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True
                    ),
                BatchNorm2d(inter_channel, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),

                nn.Upsample(scale_factor=2, mode='bilinear'),
                exit_conv(
                    in_channels=inter_channel,
                    out_channels=inter_channel,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True
                    ),
                BatchNorm2d(inter_channel, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),

                nn.Upsample(scale_factor=2, mode='bilinear'),
                exit_conv(
                    in_channels=inter_channel,
                    out_channels=config.DATASET.NUM_CLASSES,
                    kernel_size=config.EXIT.FINAL_CONV_KERNEL,
                    stride=1,
                    padding=1 if config.EXIT.FINAL_CONV_KERNEL == 3 else 0,
                    bias=True, 
                    )
            ]

        exit_layer = nn.Sequential(*exit_layer)

        return exit_layer

    def get_exit_input(self, x, detach=True):
        interpolated_list = [x[0]]
        x0_h, x0_w = x[0].size(2), x[0].size(3)

        for i in range(1, len(x)):
            interpolated_list.append(F.upsample(x[i], size=(x0_h, x0_w), mode='bilinear'))

        ret = torch.cat(interpolated_list, 1)

        return ret.detach() if detach else ret



    def init_weights(self, pretrained='', load_stage=1):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if os.path.isfile(pretrained) and load_stage == 0:
            pretrained_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
        elif os.path.isfile(pretrained) and load_stage == 1:
            pretrained_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()
            pretrained_dict = {k[len('model.'):]: v for k, v in pretrained_dict.items() if k[len('model.'):] in model_dict.keys()}
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if not k.startswith('exit')}

        elif os.path.isfile(pretrained) and load_stage == 2:
            pretrained_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()
            pretrained_dict = {k[len('model.'):]: v for k, v in pretrained_dict.items() if k[len('model.'):] in model_dict.keys()}

        logger.info('loading stage: {}, loading {} dict keys'.format(load_stage, len(pretrained_dict)))
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)


class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm, self).__init__()
    def forward(self, x):
        return F.normalize(x, p=2, dim=0)


class TemperatureScaling(nn.Module):
    def __init__(self, channel_wise, location_wise):
        super(TemperatureScaling, self).__init__()

        self.channel_wise = channel_wise
        self.location_wise = location_wise
        self.shift = 0.5413

    def forward(self, x):
        pass

class TemperatureScalingFixed(TemperatureScaling):
    def __init__(self, channel_wise=False, location_wise=False, num_channels=0):

        super(TemperatureScalingFixed, self).__init__(channel_wise=channel_wise, location_wise=location_wise)
        self.num_channels = num_channels

        assert (not self.location_wise)

        if channel_wise:
            self.t_vector = nn.Parameter(torch.zeros(num_channels), requires_grad=True)
        else:
            self.t = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x):

        if self.channel_wise:
            positive_t_vector = F.softplus(self.t_vector + self.shift)
            out = x * positive_t_vector[None, :, None, None]
        else:
            positive_t = F.softplus(self.t + self.shift)
            out = x * positive_t
        return out

class TemperatureScalingPredicted(TemperatureScaling):
    def __init__(self, channel_wise=False, location_wise=False, in_channels=0, layer_type='conv1'):
        super(TemperatureScalingPredicted, self).__init__(channel_wise=channel_wise, location_wise=location_wise)
        assert self.location_wise

        self.in_channels = in_channels
        self.layer_type = layer_type

        if self.layer_type == 'conv1':
            self.layer = used_conv(in_channels, 1, kernel_size=1, padding=0)
        elif self.layer_type == 'conv3':
            self.layer = used_conv(in_channels, 1, kernel_size=3, padding=1)
        elif self.layer_type == 'default_exit':
            self.layer = nn.Sequential(
            used_conv(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=1,
                stride=1,
                padding=0),
            BatchNorm2d(in_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True), 
            used_conv(
                in_channels=in_channels,
                out_channels=1,
                kernel_size=1,
                stride=1,
                padding=0)
        )
        else:
            raise NotImplementedError('TemperatureScalingPredicted layer type {} not implemented!'.format(self.layer_type))

    def forward(self, x):
        logits = x[0]
        features = x[1]
        self.t_map = self.layer(features) * 1.0
        self.positive_t_map = F.softplus(self.t_map + self.shift)
        return logits * self.positive_t_map


def get_seg_model(cfg, **kwargs):
    model = HighResolutionNet(cfg, **kwargs)
    model.init_weights(cfg.MODEL.PRETRAINED, cfg.MODEL.LOAD_STAGE)

    return model

if __name__ == '__main__':
    from config import config
    from config import update_config
    import argparse
    import torch.backends.cudnn as cudnn

    def parse_args():
        parser = argparse.ArgumentParser(description='Train segmentation network')
        
        parser.add_argument('--cfg',
                            help='experiment configure file name',
                            type=str, default='experiments/cityscapes/seg_hrnet_ee_0715_mask.yaml')
        parser.add_argument('opts',
                            help="Modify config options using the command-line",
                            default=None,
                            nargs=argparse.REMAINDER)
        args = parser.parse_args()
        update_config(config, args)
        return args

    args = parse_args()
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    model = eval('get_seg_model')(config)
    model = nn.DataParallel(model, device_ids=[0]).cuda()

    for i in range(20):
        print(i)
        dump_input = torch.rand(
            (1, 3, config.TRAIN.IMAGE_SIZE[1]//4, config.TRAIN.IMAGE_SIZE[0]//4)
        )
        out = model(dump_input)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(count_parameters(model))    

