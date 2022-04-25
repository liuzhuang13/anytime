import logging
import os
import time

import numpy as np
import numpy.ma as ma
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn import functional as F

from utils.utils import AverageMeter
from utils.utils import get_confusion_matrix_gpu
from utils.utils import adjust_learning_rate
from utils.utils import get_world_size, get_rank
from utils.modelsummary import get_model_summary

import pdb
from PIL import Image
import cv2
import time

def reduce_tensor(inp):
    world_size = get_world_size()
    if world_size < 2:
        return inp
    with torch.no_grad():
        reduced_inp = inp
        dist.reduce(reduced_inp, dst=0)
    return reduced_inp


def train_ee(config, epoch, num_epoch, epoch_iters, base_lr, num_iters,
         trainloader, optimizer, model, writer_dict, device):
    
    model.train()
    torch.manual_seed(get_rank() + epoch * 123)

    if config.TRAIN.EE_ONLY or config.TRAIN.ALLE_ONLY:
        model.eval()
        model.module.model.exit1.train()
        model.module.model.exit2.train()
        model.module.model.exit3.train()
    if config.TRAIN.ALLE_ONLY:
        model.module.model.last_layer.train()


    data_time = AverageMeter()
    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    
    tic_data = time.time()
    tic = time.time()
    tic_total = time.time()
    cur_iters = epoch*epoch_iters
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']
    rank = get_rank()
    world_size = get_world_size()

    for i_iter, batch in enumerate(trainloader):
        data_time.update(time.time() - tic_data)


        images, labels, _, _ = batch
        images = images.to(device)
        labels = labels.long().to(device)

        losses, _ = model(images, labels)

        loss = 0
        reduced_losses = []
        for i, l in enumerate(losses):
            loss += config.MODEL.EXTRA.EE_WEIGHTS[i] * losses[i]
            reduced_losses.append(reduce_tensor(losses[i]))
        reduced_loss = reduce_tensor(loss)

        model.zero_grad()
        loss.backward()
        optimizer.step()


        ave_loss.update(reduced_loss.item())

        lr = adjust_learning_rate(optimizer,
                                  base_lr,
                                  num_iters,
                                  i_iter+cur_iters)

        batch_time.update(time.time() - tic)
        tic = time.time()


        if i_iter % config.PRINT_FREQ == 0 and rank == 0:

            print_loss = reduced_loss / world_size
            msg = 'Epoch: [{: >3d}/{}] Iter:[{: >3d}/{}], Time: {:.2f}, Data Time: {:.2f} ' \
                  'lr: {:.6f}, Loss: {:.6f}' .format(
                      epoch, num_epoch, i_iter, epoch_iters, 
                      batch_time.average(), data_time.average(), lr, print_loss)
            logging.info(msg)
            
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', print_loss, global_steps)

            writer.add_scalars('exit_train_loss', {
                'exit1': reduced_losses[0].item() / world_size,
                'exit2': reduced_losses[1].item() / world_size,
                'exit3': reduced_losses[2].item() / world_size,
                'exit4': reduced_losses[3].item() / world_size,
                }, 
            global_steps)

            writer_dict['train_global_steps'] += 1

        tic_data = time.time()

    train_time = time.time() - tic_total

    if rank == 0:
        logging.info(f'Train time:{train_time}s')

def validate_ee(config, testloader, model, writer_dict, device):

    torch.manual_seed(get_rank())
    
    tic_data = time.time()
    tic = time.time()
    tic_total = time.time()
    rank = get_rank()
    world_size = get_world_size()
    model.eval()

    data_time = AverageMeter()
    batch_time = AverageMeter()
    ave_loss = AverageMeter()

    num_exits = len(config.MODEL.EXTRA.EE_WEIGHTS)

    ave_losses = [AverageMeter() for i in range(num_exits)]

    confusion_matrices = [np.zeros((config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES)) for i in range(num_exits)]


    with torch.no_grad():
        for i_iter, batch in enumerate(testloader):
            data_time.update(time.time() - tic_data)

            image, label, _, _ = batch
            size = label.size()
            image = image.to(device)
            label = label.long().to(device)

            losses, preds = model(image, label)
            
            for i, pred in enumerate(preds):
                if pred.size()[-2] != size[-2] or pred.size()[-1] != size[-1]:
                    pred = F.upsample(pred, (size[-2], size[-1]), 
                                       mode='bilinear')

                confusion_matrices[i] += get_confusion_matrix_gpu(
                    label,
                    pred,
                    size,
                    config.DATASET.NUM_CLASSES,
                    config.TRAIN.IGNORE_LABEL)

            loss = 0
            reduced_losses = []
            for i, l in enumerate(losses):
                loss += config.MODEL.EXTRA.EE_WEIGHTS[i] * losses[i]
                reduced_losses.append(reduce_tensor(losses[i]))
                ave_losses[i].update(reduced_losses[i].item())

            reduced_loss = reduce_tensor(loss)
            ave_loss.update(reduced_loss.item())

            batch_time.update(time.time() - tic)
            tic = time.time()

            tic_data = time.time()

            if i_iter % config.PRINT_FREQ == 0 and rank == 0:
                print_loss = ave_loss.average() / world_size 
                msg = 'Iter:[{: >3d}/{}], Time: {:.2f}, Data Time: {:.2f} ' \
                       'Loss: {:.6f}' .format(
                          i_iter, len(testloader), batch_time.average(), data_time.average(), print_loss)
                logging.info(msg)


    results = []
    for i, confusion_matrix in enumerate(confusion_matrices):
    
        confusion_matrix = torch.from_numpy(confusion_matrix).to(device)
        reduced_confusion_matrix = reduce_tensor(confusion_matrix)
        confusion_matrix = reduced_confusion_matrix.cpu().numpy()

        pos = confusion_matrix.sum(1)
        res = confusion_matrix.sum(0)
        tp = np.diag(confusion_matrix)
        pixel_acc = tp.sum()/pos.sum()
        mean_acc = (tp/np.maximum(1.0, pos)).mean()
        IoU_array = (tp / np.maximum(1.0, pos + res - tp))
        mean_IoU = IoU_array.mean()

        results.append((mean_IoU, IoU_array, pixel_acc, mean_acc))

    val_time = time.time() - tic_total

    if rank == 0:
        logging.info(f'Validation time:{val_time}s')
        mean_IoUs = [result[0] for result in results]
        mean_IoUs.append(np.mean(mean_IoUs))
        print_result = '\t'.join(['{:.2f}'.format(m*100) for m in mean_IoUs])
        logging.info(f'mean_IoUs: {print_result}')

        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('valid_loss', print_loss, global_steps)

        writer.add_scalars('exit_valid_loss', {
            'exit1': ave_losses[0].average() / world_size,
            'exit2': ave_losses[1].average() / world_size,
            'exit3': ave_losses[2].average() / world_size,
            'exit4': ave_losses[3].average() / world_size,
            }, 
        global_steps)

        writer.add_scalars('valid_mIoUs',
            {f'valid_mIoU{i+1}': results[i][0] for i in range(num_exits)}, 
            global_steps
            )
        writer_dict['valid_global_steps'] += 1

    return results


VIS_T = False
VIS = False
VIS_CONF = False
TIMING = True

def testval_ee(config, test_dataset, testloader, model, 
        sv_dir='', sv_pred=False):
    model.eval()
    torch.manual_seed(get_rank())
    num_exits = len(config.MODEL.EXTRA.EE_WEIGHTS)

    confusion_matrices = [np.zeros((config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES)) for i in range(num_exits)]

    total_time = 0

    with torch.no_grad():
        for index, batch in enumerate(tqdm(testloader)):
            image, label, _, name = batch
            if config.PYRAMID_TEST.USE:
                image = F.interpolate(image, (config.PYRAMID_TEST.SIZE//2, config.PYRAMID_TEST.SIZE), mode='bilinear')

            size = label.size()

            if TIMING:
                start = time.time()
                torch.cuda.synchronize()
            preds = model(image)

            if TIMING:
                torch.cuda.synchronize()
                total_time += time.time() - start
            
            for i, pred in enumerate(preds):
                if pred.size()[-2] != size[-2] or pred.size()[-1] != size[-1]:
                    original_logits = pred
                    pred = F.upsample(pred, (size[-2], size[-1]), 
                                       mode='bilinear')

                confusion_matrices[i] += get_confusion_matrix_gpu(
                    label,
                    pred,
                    size,
                    config.DATASET.NUM_CLASSES,
                    config.TRAIN.IGNORE_LABEL)

                if sv_pred and index % 20 == 0 and VIS:
                    print("Saving ... ", name)
                    sv_path = os.path.join(sv_dir, f'test_val_results/{i+1}')
                    os.makedirs(sv_path, exist_ok=True)
                    test_dataset.save_pred(pred, sv_path, name)

                    if VIS_T or VIS_CONF:
                        def save_float_img(t, sv_path, name, normalize=False):
                            os.makedirs(sv_path, exist_ok=True)
                            if normalize:
                                t = t/t.max()
                            torch.save(t, os.path.join(sv_path, name[0]+'.pth'))
                            t = t[0][0]
                            t = t.cpu().numpy().copy()
                            np.save(os.path.join(sv_path, name[0]+'.npy'), t)
                            cv2.imwrite(os.path.join(sv_path, name[0]+'.png'), t*255)

                        def save_long_img(t, sv_path, name):
                            os.makedirs(sv_path, exist_ok=True)
                            t = t[0][0]
                            t = t.cpu().numpy().copy()
                            cv2.imwrite(os.path.join(sv_path, name[0]+'.png'), t)

                        def save_tensor(t, sv_path, name):
                            os.makedirs(sv_path, exist_ok=True)
                            torch.save(t, os.path.join(sv_path, name[0]+'.pth'))


                    if VIS_CONF:

                        out = F.softmax(original_logits, dim=1)

                        sv_path = os.path.join(sv_dir, f'test_val_original_conf/{i+1}')
                        original_conf_map, _ = out.max(dim=1)
                        save_float_img(original_conf_map.unsqueeze(0), sv_path, name, normalize=False)

                        sv_path = os.path.join(sv_dir, f'test_val_original_pred/{i+1}')
                        max_index = torch.max(out, dim=1)[1]
                        save_long_img(max_index.unsqueeze(0), sv_path, name)

                        sv_path = os.path.join(sv_dir, f'test_val_original_logits/{i+1}')
                        save_tensor(original_logits, sv_path, name)

                        sv_path = os.path.join(sv_dir, f'test_val_original_results/{i+1}')
                        os.makedirs(sv_path, exist_ok=True)
                        test_dataset.save_pred(original_logits, sv_path, name)

                        if hasattr(model.module, 'mask_dict'):
                            sv_path = os.path.join(sv_dir, f'test_val_masks/')
                            os.makedirs(sv_path, exist_ok=True)
                            torch.save(model.module.mask_dict, os.path.join(sv_path, name[0]+'.pth'))

                        if i == 0:
                            sv_path = os.path.join(sv_dir, f'test_val_gt/')
                            save_long_img(label.unsqueeze(0), sv_path, name)
                if index % 100 == 0:
                    logging.info(f'processing: {index} images with exit {i}')
                    pos = confusion_matrices[i].sum(1)
                    res = confusion_matrices[i].sum(0)
                    tp = np.diag(confusion_matrices[i])
                    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
                    mean_IoU = IoU_array.mean()
                    logging.info('mIoU: %.4f' % (mean_IoU))

    results = []
    for i, confusion_matrix in enumerate(confusion_matrices):
        pos = confusion_matrix.sum(1)
        res = confusion_matrix.sum(0)
        tp = np.diag(confusion_matrix)
        pixel_acc = tp.sum()/pos.sum()
        mean_acc = (tp/np.maximum(1.0, pos)).mean()
        IoU_array = (tp / np.maximum(1.0, pos + res - tp))
        mean_IoU = IoU_array.mean()

        results.append((mean_IoU, IoU_array, pixel_acc, mean_acc))

    if TIMING:
        print("Total_time", total_time)

    return results


def testval_ee_class(config, test_dataset, testloader, model, 
        sv_dir='', sv_pred=False):
    model.eval()
    torch.manual_seed(get_rank())
    num_exits = len(config.MODEL.EXTRA.EE_WEIGHTS)

    confusion_matrices = [np.zeros((config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES)) for i in range(num_exits)]

    total_time = 0

    with torch.no_grad():
        for index, batch in enumerate(tqdm(testloader)):
            image, label, _, name = batch

            size = label.size()
            preds = model(image)
            
            for i, pred in enumerate(preds):
                if pred.size()[-2] != size[-2] or pred.size()[-1] != size[-1]:
                    original_logits = pred
                    pred = F.upsample(pred, (size[-2], size[-1]), 
                                       mode='bilinear')

                confusion_matrices[i] += get_confusion_matrix_gpu(
                    label,
                    pred,
                    size,
                    config.DATASET.NUM_CLASSES,
                    config.TRAIN.IGNORE_LABEL)

                if sv_pred and index % 20 == 0 and VIS:
                    print("Saving ... ", name)
                    sv_path = os.path.join(sv_dir, f'test_val_results/{i+1}')
                    os.makedirs(sv_path, exist_ok=True)
                    test_dataset.save_pred(pred, sv_path, name)

                    if VIS_T or VIS_CONF:
                        def save_float_img(t, sv_path, name, normalize=False):
                            os.makedirs(sv_path, exist_ok=True)
                            if normalize:
                                t = t/t.max()
                            torch.save(t, os.path.join(sv_path, name[0]+'.pth'))
                            t = t[0][0]
                            t = t.cpu().numpy().copy()
                            np.save(os.path.join(sv_path, name[0]+'.npy'), t)
                            cv2.imwrite(os.path.join(sv_path, name[0]+'.png'), t*255)

                        def save_long_img(t, sv_path, name):
                            os.makedirs(sv_path, exist_ok=True)
                            t = t[0][0]
                            t = t.cpu().numpy().copy()
                            cv2.imwrite(os.path.join(sv_path, name[0]+'.png'), t)

                        def save_tensor(t, sv_path, name):
                            os.makedirs(sv_path, exist_ok=True)
                            torch.save(t, os.path.join(sv_path, name[0]+'.pth'))
                    if VIS_CONF:
                        out = F.softmax(original_logits, dim=1)

                        sv_path = os.path.join(sv_dir, f'test_val_original_conf/{i+1}')
                        original_conf_map, _ = out.max(dim=1)
                        save_float_img(original_conf_map.unsqueeze(0), sv_path, name, normalize=False)

                        sv_path = os.path.join(sv_dir, f'test_val_original_pred/{i+1}')
                        max_index = torch.max(out, dim=1)[1]
                        save_long_img(max_index.unsqueeze(0), sv_path, name)

                        sv_path = os.path.join(sv_dir, f'test_val_original_logits/{i+1}')
                        save_tensor(original_logits, sv_path, name)

                        sv_path = os.path.join(sv_dir, f'test_val_original_results/{i+1}')
                        os.makedirs(sv_path, exist_ok=True)
                        test_dataset.save_pred(original_logits, sv_path, name)

                        if hasattr(model.module, 'mask_dict'):
                            sv_path = os.path.join(sv_dir, f'test_val_masks/')
                            os.makedirs(sv_path, exist_ok=True)
                            torch.save(model.module.mask_dict, os.path.join(sv_path, name[0]+'.pth'))

                        if i == 0:
                            sv_path = os.path.join(sv_dir, f'test_val_gt/')
                            save_long_img(label.unsqueeze(0), sv_path, name)

                if index % 100 == 0:
                    logging.info(f'processing: {index} images with exit {i}')
                    pos = confusion_matrices[i].sum(1)
                    res = confusion_matrices[i].sum(0)
                    tp = np.diag(confusion_matrices[i])
                    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
                    mean_IoU = IoU_array.mean()
                    logging.info('mIoU: %.4f' % (mean_IoU))

    results = []
    for i, confusion_matrix in enumerate(confusion_matrices):
        pos = confusion_matrix.sum(1)
        res = confusion_matrix.sum(0)
        tp = np.diag(confusion_matrix)
        pixel_acc = tp.sum()/pos.sum()
        mean_acc = (tp/np.maximum(1.0, pos)).mean()
        IoU_array = (tp / np.maximum(1.0, pos + res - tp))
        mean_IoU = IoU_array.mean()

        results.append((mean_IoU, IoU_array, pixel_acc, mean_acc))

    if TIMING:
        print("Total_time", total_time)

    return results
def testval_ee_profiling(config, test_dataset, testloader, model, 
        sv_dir='', sv_pred=False):
    model.eval()
    torch.manual_seed(get_rank())
    num_exits = len(config.MODEL.EXTRA.EE_WEIGHTS)
    total_time = 0

    gflops = []
    with torch.no_grad():
        for index, batch in enumerate(tqdm(testloader)):
            image, label, _, name = batch
            if config.PYRAMID_TEST.USE:
                image = F.interpolate(image, (config.PYRAMID_TEST.SIZE, config.PYRAMID_TEST.SIZE//2), mode='bilinear')
            stats = {}
            saved_stats = {}

            for i in range(4):
                setattr(model.module, f"stop{i+1}", "anY_RanDOM_ThiNg")
                summary, stats[i+1] = get_model_summary(model, image, verbose=False)
                delattr(model.module, f"stop{i+1}")

            saved_stats['params'] = [stats[i+1]['params'] for i in range(4)]
            saved_stats['flops'] = [stats[i+1]['flops'] for i in range(4)]
            saved_stats['counts'] = [stats[i+1]['counts'] for i in range(4)]
            saved_stats['Gflops'] = [f/(1024**3) for f in saved_stats['flops']]
            saved_stats['Mparams'] = [f/(10**6) for f in saved_stats['params']]
            gflops.append(saved_stats['Gflops'])

    final_stats = saved_stats
    final_stats['Gflops'] = []
    for i in range(4):
        final_stats['Gflops'].append(np.mean([x[i] for x in gflops]))
    final_stats['Gflops_mean'] = np.mean(final_stats['Gflops'])
    return final_stats

def testval_ee_profiling_actual(config, test_dataset, testloader, model, 
        sv_dir='', sv_pred=False):
    model.eval()
    torch.manual_seed(get_rank())
    num_exits = len(config.MODEL.EXTRA.EE_WEIGHTS)
    total_time = 0

    stats = {}
    stats['time'] = {}
    times = []

    with torch.no_grad():
        for index, batch in enumerate(tqdm(testloader)):
            image, label, _, name = batch
            t = []
            for i in range(4):
                if isinstance(model, nn.DataParallel):
                    setattr(model.module, f"stop{i+1}", "anY_RanDOM_ThiNg")
                else:
                    setattr(model, f"stop{i+1}", "anY_RanDOM_ThiNg")

                torch.cuda.synchronize()
                start = time.time()
                out = model(image)
                torch.cuda.synchronize()
                t.append(time.time() - start)

                if isinstance(model, nn.DataParallel):
                    delattr(model.module, f"stop{i+1}")
                else:
                    delattr(model, f"stop{i+1}")

            if index > 5:
                times.append(t)
            if index > 20:
                break

            print(t)
    for i in range(4):
        stats['time'][i] = np.mean([t[i] for t in times])
        print(stats)   
    return stats
