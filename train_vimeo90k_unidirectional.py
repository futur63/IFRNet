# IFRNet_U Unidirectional Training Script
# 사용법 예시:
# python IFRNet/train_vimeo90k_unidirectional.py --epochs 10 --batch_size 4

import os
import math
import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import Vimeo90K_Train_Dataset, Vimeo90K_Test_Dataset
from metric import calculate_psnr, calculate_ssim
from utils import AverageMeter, warp
import logging
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
import cv2


def get_lr(args, iters):
    ratio = 0.5 * (1.0 + np.cos(iters / (args.epochs * args.iters_per_epoch) * math.pi))
    lr = (args.lr_start - args.lr_end) * ratio + args.lr_end
    return lr


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def flow2rgb(flow_map_np):
    h, w, _ = flow_map_np.shape
    rgb_map = np.ones((h, w, 3)).astype(np.float32)
    normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max() + 1e-6)
    rgb_map[:, :, 0] += normalized_flow_map[:, :, 0]
    rgb_map[:, :, 1] -= 0.5 * (normalized_flow_map[:, :, 0] + normalized_flow_map[:, :, 1])
    rgb_map[:, :, 2] += normalized_flow_map[:, :, 1]
    return rgb_map.clip(0, 1)


def train(args, model):
    print('Single GPU Unidirectional Training IFRNet_U')

    os.makedirs(args.log_path, exist_ok=True)
    log_path = os.path.join(args.log_path, time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime()))
    os.makedirs(log_path, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel('INFO')
    BASIC_FORMAT = '%(asctime)s:%(levelname)s:%(message)s'
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
    chlr = logging.StreamHandler()
    chlr.setFormatter(formatter)
    chlr.setLevel('INFO')
    fhlr = logging.FileHandler(os.path.join(log_path, 'train.log'))
    fhlr.setFormatter(formatter)
    logger.addHandler(chlr)
    logger.addHandler(fhlr)
    logger.info(args)

    try:
        print("Loading training dataset...")
        dataset_train = Vimeo90K_Train_Dataset(dataset_dir='../Dataset/vimeo_triplet', augment=True)
        print(f"Training dataset loaded: {len(dataset_train)} samples")
        dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, drop_last=True, shuffle=True)
        args.iters_per_epoch = dataloader_train.__len__()
        iters = args.resume_epoch * args.iters_per_epoch
        print(f"Training dataloader created: {args.iters_per_epoch} iterations per epoch")
        
        print("Loading validation dataset...")
        dataset_val = Vimeo90K_Test_Dataset(dataset_dir='../Dataset/vimeo_triplet')
        print(f"Validation dataset loaded: {len(dataset_val)} samples")
        dataloader_val = DataLoader(dataset_val, batch_size=8, num_workers=8, pin_memory=True, shuffle=False, drop_last=True)
        print("Validation dataloader created")
    except Exception as e:
        logger.error(f"Dataset load failed: {e}")
        logger.error("Please make sure vimeo_triplet dataset is in the current directory")
        return

    optimizer = optim.AdamW(model.parameters(), lr=args.lr_start, weight_decay=0)

    time_stamp = time.time()
    avg_05 = AverageMeter()
    avg_1 = AverageMeter()
    avg_total = AverageMeter()
    best_psnr = 0.0

    writer = SummaryWriter(log_dir=log_path)

    for epoch in range(args.resume_epoch, args.epochs):
        print(f"Starting epoch {epoch+1}/{args.epochs}")
        for i, data in enumerate(dataloader_train):
            for l in range(len(data)):
                data[l] = data[l].to(args.device)
            img_0, img_2, img_1, flow, embt = data
            img_1_gt = img_1.clone() 

            data_time_interval = time.time() - time_stamp
            time_stamp = time.time()

            lr = get_lr(args, iters)
            set_lr(optimizer, lr)

            optimizer.zero_grad()

            img1_pred, img2_pred, flow_0_1, flow_1_2, mask0, mask1, residual0, residual2, loss \
                = model(img_0, img_1, embt, img_1_gt, img_2)

            loss.backward()
            optimizer.step()

            avg_total.update(loss.cpu().data)
            train_time_interval = time.time() - time_stamp

            if (iters+1) % 100 == 0:
                logger.info('epoch:{}/{} iter:{}/{} time:{:.2f}+{:.2f} lr:{:.5e} loss_total:{:.4e}'.format(epoch+1, args.epochs, iters+1, args.epochs * args.iters_per_epoch, data_time_interval, train_time_interval, lr, avg_total.avg))
                avg_total.reset()
                # TensorBoard 시각화 (첫 번째 배치만)
                idx = 0
                writer.add_scalar('train/loss', loss.item(), iters)
                writer.add_scalar('train/learning_rate', lr, iters)
                writer.add_image('input/img_0', img_0[idx], iters, dataformats='CHW')
                writer.add_image('input/img_1', img_1[idx], iters, dataformats='CHW')
                writer.add_image('input/img_2', img_2[idx], iters, dataformats='CHW')
                writer.add_image('pred/img1_pred', img1_pred[idx], iters, dataformats='CHW')
                writer.add_image('pred/img2_pred', img2_pred[idx], iters, dataformats='CHW')
              
                # --- 추가: warp-only, residual-only 시각화 ---
                img1_warp_only = warp(img_0, flow_0_1)
                img1_residual_only = img_0 + residual0
                img2_warp_only = warp(img_1, flow_1_2)
                img2_residual_only = img_1 + residual2

                img1_warp_only = torch.clamp(img1_warp_only, 0, 1)
                img1_residual_only = torch.clamp(img1_residual_only, 0, 1)
                img2_warp_only = torch.clamp(img2_warp_only, 0, 1)
                img2_residual_only = torch.clamp(img2_residual_only, 0, 1)

                writer.add_image('ablation/img1_warp_only', img1_warp_only[idx], iters, dataformats='CHW')
                writer.add_image('ablation/img1_residual_only', img1_residual_only[idx], iters, dataformats='CHW')
                writer.add_image('ablation/img2_warp_only', img2_warp_only[idx], iters, dataformats='CHW')
                writer.add_image('ablation/img2_residual_only', img2_residual_only[idx], iters, dataformats='CHW')
                # --- END 추가 ---
                # flow 시각화
                flow_np_0_1 = flow_0_1[idx].detach().cpu().numpy().transpose(1,2,0)  # (H,W,2)
                flow_np_1_2 = flow_1_2[idx].detach().cpu().numpy().transpose(1,2,0)
                flow_img_0_1 = flow2rgb(flow_np_0_1)
                flow_img_1_2 = flow2rgb(flow_np_1_2)
                writer.add_image('flow/flow_0_1', flow_img_0_1.transpose(2,0,1), iters, dataformats='CHW')
                writer.add_image('flow/flow_1_2', flow_img_1_2.transpose(2,0,1), iters, dataformats='CHW')

            iters += 1
            time_stamp = time.time()

        if (epoch+1) % args.eval_interval == 0:
            print(f"Evaluating epoch {epoch+1}")
            psnr = evaluate(args, model, dataloader_val, epoch, logger, writer, iters)
            if psnr > best_psnr:
                best_psnr = psnr
                torch.save(model.state_dict(), '{}/{}_{}.pth'.format(log_path, args.model_name, 'best'))
            torch.save(model.state_dict(), '{}/{}_{}.pth'.format(log_path, args.model_name, 'latest'))

    writer.close()


def evaluate(args, model, dataloader_val, epoch, logger, writer, global_step):
    loss_list = []
    psnr_list = []
    time_stamp = time.time()
    for i, data in enumerate(dataloader_val):
        for l in range(len(data)):
            data[l] = data[l].to(args.device)
        img_0, img_2, img_1, flow, embt = data

        with torch.no_grad():
            img1_pred, img2_pred, flow_0_1, flow_1_2, mask0, mask1, residual0, residual2, loss = model(img_0, img_1, embt, img_1, img_2)
            loss = (img1_pred - img_1).abs().mean()

        loss_list.append(loss.cpu().numpy())

        for j in range(img_0.shape[0]):
            psnr = calculate_psnr(img1_pred[j].unsqueeze(0), img_1[j].unsqueeze(0)).cpu().data
            psnr_list.append(psnr)

    eval_time_interval = time.time() - time_stamp
    logger.info('eval epoch:{}/{} time:{:.2f} loss_total:{:.4e} psnr:{:.3f}'.format(epoch+1, args.epochs, eval_time_interval, np.array(loss_list).mean(), np.array(psnr_list).mean()))
    writer.add_scalar('eval/loss', np.array(loss_list).mean(), global_step)
    writer.add_scalar('eval/psnr', np.array(psnr_list).mean(), global_step)
    return np.array(psnr_list).mean()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IFRNet_U Unidirectional Single GPU Training')
    parser.add_argument('--model_name', default='IFRNet_U', type=str, help='IFRNet_U')
    parser.add_argument('--epochs', default=31, type=int)
    parser.add_argument('--eval_interval', default=1, type=int)
    parser.add_argument('--batch_size', default=12, type=int)
    parser.add_argument('--lr_start', default=1e-4, type=float)
    parser.add_argument('--lr_end', default=1e-5, type=float)
    parser.add_argument('--log_path', default='checkpoint', type=str)
    parser.add_argument('--resume_epoch', default=0, type=int)
    parser.add_argument('--resume_path', default=None, type=str)
    args = parser.parse_args()

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {args.device}")

    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    from models.IFRNet_U import Model

    args.log_path = args.log_path + '/' + args.model_name
    args.num_workers = args.batch_size

    model = Model().to(args.device)
    
    if args.resume_epoch != 0:
        model.load_state_dict(torch.load(args.resume_path, map_location='cpu'))
    
    train(args, model) 