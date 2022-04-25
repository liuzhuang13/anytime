from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
from collections import namedtuple

import torch
import torch.nn as nn

def get_model_summary(model, *input_tensors, item_length=26, verbose=False):
    summary = []

    ModuleDetails = namedtuple(
        "Layer", ["name", "input_size", "output_size", "num_parameters", "multiply_adds", "num_param_counts"])
    hooks = []
    layer_instances = {}

    def add_hooks(module):

        def hook(module, input, output):
            class_name = str(module.__class__.__name__)

            instance_index = 1
            if class_name not in layer_instances:
                layer_instances[class_name] = instance_index
            else:
                instance_index = layer_instances[class_name] + 1
                layer_instances[class_name] = instance_index

            layer_name = class_name + "_" + str(instance_index)

            params = 0
            counts = 0
            if class_name.find("Conv") != -1 or class_name.find("BatchNorm") != -1 or \
               class_name.find("Linear") != -1 or class_name.find('conv') != -1 or class_name.find('Temp') != -1:
                for param_ in module.parameters():
                    if param_.requires_grad:
                        params += param_.view(-1).size(0)
                        counts += 1

            flops = "Not Available"

            if (class_name.find("ConvTranspose2d") != -1) and hasattr(module, "weight"): 
                flops = (
                    torch.prod(
                        torch.LongTensor(list(module.weight.data.size()))) *
                    torch.prod(
                        torch.LongTensor(list(input[0].size())[2:]))).item()


            elif (class_name.find("Conv") != -1) and hasattr(module, "weight"):
                flops = (
                    torch.prod(
                        torch.LongTensor(list(module.weight.data.size()))) *
                    torch.prod(
                        torch.LongTensor(list(output.size())[2:]))).item()


            elif class_name.find("conv")!= -1:
                flops = (
                    torch.prod(
                        torch.LongTensor(list(module.weight.data.size()))) *
                    torch.prod(
                        torch.LongTensor(list(output.size())[2:]))).item()
                p = module.mask.mean().item()
                p2 = module.p
                flops = int(flops * p) 
                if module.interpolate == 'rbf' or module.interpolate == 'pooling' or module.interpolate == 'conv':
                    flops += int((1 - p) * p * output.size(1) * (module.r * module.r) * output.size(2) * output.size(3))
            elif isinstance(module, nn.Linear):
                flops = (torch.prod(torch.LongTensor(list(output.size()))) \
                         * input[0].size(1)).item()

            if isinstance(input[0], list):
                input = input[0]
            if isinstance(output, list):
                output = output[0]

            summary.append(
                ModuleDetails(
                    name=layer_name,
                    input_size=list(input[0].size()),
                    output_size=list(output.size()),
                    num_parameters=params,
                    multiply_adds=flops,
                    num_param_counts=counts,)
            )

        if not isinstance(module, nn.ModuleList) \
           and not isinstance(module, nn.Sequential) \
           and module != model:
            hooks.append(module.register_forward_hook(hook))

    model.eval()
    model.apply(add_hooks)

    space_len = item_length

    model(*input_tensors)
    for hook in hooks:
        hook.remove()

    details = ''
    if verbose:
        details = "Model Summary" + \
            os.linesep + \
            "Name{}Input Size{}Output Size{}Parameters{}Multiply Adds (Flops){}".format(
                ' ' * (space_len - len("Name")),
                ' ' * (space_len - len("Input Size")),
                ' ' * (space_len - len("Output Size")),
                ' ' * (space_len - len("Parameters")),
                ' ' * (space_len - len("Multiply Adds (Flops)"))) \
                + os.linesep + '-' * space_len * 5 + os.linesep

    params_sum = 0
    flops_sum = 0
    counts_sum = 0
    for layer in summary:
        params_sum += layer.num_parameters
        if layer.multiply_adds != "Not Available":
            flops_sum += layer.multiply_adds
        counts_sum += layer.num_param_counts
        shown_flops = layer.multiply_adds/(1024**3) if layer.multiply_adds != 'Not Available' else 0
        if verbose:
            if shown_flops < 1:
                details += ''
            else:
                details += "{}{}{}{}{}{}{}{}{}{}".format(
                layer.name,
                ' ' * (space_len - len(layer.name)),
                layer.input_size,
                ' ' * (space_len - len(str(layer.input_size))),
                layer.output_size,
                ' ' * (space_len - len(str(layer.output_size))),
                layer.num_parameters,
                ' ' * (space_len - len(str(layer.num_parameters))),
                shown_flops,
                ' ' * (space_len - len(str(shown_flops)))) \
                + os.linesep + '-' * space_len * 5 + os.linesep

    details += os.linesep \
        + "Total Parameters: {:,}".format(params_sum) \
        + os.linesep + '-' * space_len * 5 + os.linesep
    details += "Total Multiply Adds (For Convolution and Linear Layers only): {:,} GFLOPs".format(flops_sum/(1024**3)) \
        + os.linesep + '-' * space_len * 5 + os.linesep
    details += "Number of Layers" + os.linesep
    for layer in layer_instances:
        details += "{} : {} layers   ".format(layer, layer_instances[layer])

    return details, {'params': params_sum,
                    'flops': flops_sum, 
                    'counts': counts_sum}
