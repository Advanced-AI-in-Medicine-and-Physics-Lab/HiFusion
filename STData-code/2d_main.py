# Standard library imports
import argparse
import logging
import math
import os
import random
import sys
import time
import warnings
from tqdm import tqdm

# Third-party imports
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.amp import GradScaler
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tensorboardX import SummaryWriter

# Local imports
from dataloader import ImageDataset, create_dataloaders_for_each_file
from models.hifusion import HiFusion
from models.losses import calculate_pcc, pcc_loss

warnings.filterwarnings("ignore", category=UserWarning, message="Creating a tensor from a list of numpy.ndarrays is extremely slow")
warnings.filterwarnings("ignore", category=FutureWarning, message="You are using `torch.load` with `weights_only=False`")


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='data/',
                    help='data path')
parser.add_argument('--save_path', type=str, default='save/ST_2d/test/',
                    help='model save path')
parser.add_argument('--max_epoch', type=int, default=50, help='maximum epoch number to train')
parser.add_argument('--base_lr', type=float, default=0.0003, help='maximum epoch number to train')
parser.add_argument('--lr_decay', type=float, default=0.9, help='learning rate decay')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--batch_size', type=int, default=32, help='repeat')
parser.add_argument('--rate', type=int, default=10, help='downsample rate')

args = parser.parse_args()

root = args.root_path
label_root = os.path.join(root, 'npy_information')
save_path = args.save_path
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
max_epoch = args.max_epoch
base_lr = args.base_lr
lr_decay = args.lr_decay
loss_record = 0
batch_size = args.batch_size
cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def timer(start,end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    return int(hours), int(minutes), int(seconds)

class PerImageNormalize(object):
    def __call__(self, img_tensor):
        mean = torch.mean(img_tensor, dim=(1, 2))  
        std = torch.std(img_tensor, dim=(1, 2))
        std[std == 0] = 1.0
        normalized = (img_tensor - mean.view(3, 1, 1)) / std.view(3, 1, 1)
        return normalized


if __name__ == "__main__":
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    hism_level=[1,4,49]
    hism_num = len(hism_level)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    logging.basicConfig(filename=save_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    img_transform = transforms.Compose([
                                        torchvision.transforms.RandomHorizontalFlip(),
                                        torchvision.transforms.RandomVerticalFlip(),
                                        torchvision.transforms.RandomApply(
                                            [torchvision.transforms.RandomRotation((90, 90))]),
                                        torchvision.transforms.ToTensor(),
                                        PerImageNormalize()
                                        ])
    test_img_transform = transforms.Compose([
                                        torchvision.transforms.ToTensor(),
                                        PerImageNormalize()
                                        ])

    mse_loss_fn = nn.MSELoss()
    l1_loss_fn = torch.nn.L1Loss()

    cross_mse, cross_mae, cross_pcc = [], [], []

    all_folders = sorted([folder for folder in os.listdir(label_root) if os.path.isdir(os.path.join(label_root, folder))])
    num_folders = len(all_folders)
    fold_size = num_folders // 4
    print(all_folders)
    start_time = time.time()
    for fold in range(4):
        start = fold * fold_size
        end = start + fold_size if fold < 4 - 1 else num_folders  # Handle remaining folders in the last fold

        val_folders = all_folders[start:end]
        train_folders = [folder for folder in all_folders if folder not in val_folders]

        train_files = []

        for folder in train_folders:
            folder_path = os.path.join(label_root, folder)
            train_files.extend([os.path.join(folder_path, f'{i}.npy') for i in range(1, 4)])

        val_files = []
        for folder in val_folders:
            folder_path = os.path.join(label_root, folder)
            val_files.extend([os.path.join(folder_path, f'{i}.npy') for i in range(1, 4)])

        train_dataloaders = create_dataloaders_for_each_file(train_files, root=root, batch_size=batch_size, transform=img_transform, shuffle=True)
        test_dataloader = create_dataloaders_for_each_file(val_files, root=root, batch_size=batch_size, transform=test_img_transform, shuffle=False)

        dl_lens = [len(train_dataloaders[dl_name]) for dl_name in list(train_dataloaders.keys())]
        print(f'--------------- max iterations per epoch:{sum(dl_lens)} ---------------')

        iter_num = 0
        lr_ = base_lr
        best_mse = 10000

        writer = SummaryWriter(save_path + '/log')

        model = HiFusion(hism_level=hism_level).cuda()

        optimizer = optim.Adam(model.parameters(), lr=base_lr, betas=(0.9, 0.999), weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=1e-6)
        scaler = GradScaler()
        for epoch_num in tqdm(range(max_epoch), ncols=70):
            model.train()

            file_names = list(train_dataloaders.keys())
            random.shuffle(file_names)
            epoch_loss = 0
            epoch_iter = 0
        
            for file_name in file_names:
                tmp_loader = train_dataloaders[file_name]

                print(f"Processing file: {file_name}")
                tmp_name = file_name.split('/')[-1][:-4]

                for spot_images,region_images,labels in tmp_loader:
                        
                    region_images = region_images.cuda()
                    spot_images = spot_images.cuda()
                    labels = labels.cuda()

                    with torch.autocast("cuda", enabled=True):
                        preds, feats = model(region_images, spot_images)

                        w = 1/hism_num
                        feature_align_loss = 0
                        gene_pred_loss = mse_loss_fn(preds[-1], labels)
                        for i in range(hism_num):
                            gene_pred_loss += w*mse_loss_fn(preds[i], labels)
                        feature_align_loss = 0.5 * (l1_loss_fn(feats[0],feats[1])+l1_loss_fn(feats[0],feats[2]))

                        loss = gene_pred_loss + feature_align_loss
                        
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    iter_num += 1
                    epoch_loss += loss.item()
                    epoch_iter += 1
                    writer.add_scalar('lr', optimizer.param_groups[0]['lr'], iter_num)
                    
                    logging.info(
                        'iteration %d :loss: %5f, gene:%5f, align:%5f, lr: %6f' %
                        (iter_num, loss, gene_pred_loss, feature_align_loss, optimizer.param_groups[0]['lr']))

            writer.add_scalar(f'loss/{fold+1}_loss', epoch_loss/epoch_iter, epoch_num)
            scheduler.step()

            if epoch_num % 1 == 0:
                model.eval()
                val_mse = 0.0
                val_pcc = 0.0
                val_mae = 0.0

                total_samples = 0

                with torch.no_grad():
                    for file_name, tmp_loader in test_dataloader.items():
                        print(f"Validtaion - Processing file: {file_name}")
                        tmp_name = file_name.split('/')[-1][:-4]

                        for region_images, spot_images, labels in tmp_loader:
                            region_images = region_images.cuda()
                            spot_images = spot_images.cuda()
                            labels = labels.cuda()
                            preds,_ = model(region_images, spot_images)
                            pred = preds[-1]
                            mse_loss = mse_loss_fn(pred, labels)

                            val_mse += mse_loss.item() * spot_images.size(0)
                            total_samples += spot_images.size(0)

                            val_mae += np.mean(np.abs(pred.cpu().numpy() - labels.cpu().numpy())) * spot_images.size(0)
                            val_pcc += calculate_pcc(pred, labels.cuda()) * spot_images.size(0)

                    writer.add_scalar(f'validation/{fold+1}_mse', val_mse / total_samples, epoch_num)

                    if val_mse / total_samples < best_mse:
                        best_mse = val_mse / total_samples
                        best_mae = val_mae / total_samples
                        best_pcc = val_pcc / total_samples
                        
                        torch.save(model.state_dict(), os.path.join(save_path, f"model_{fold}.pth"))
                        print(f'Best model saved to {os.path.join(save_path, f"model_{fold}.pth")}')
                    
                    print('-'*64)
                    print(f"Validation MSE:{val_mse / total_samples} | MAE:{val_mae / total_samples} | PCC:{val_pcc / total_samples}")
        h, m, s = timer(start_time, time.time())
        print('Split-{} Training Completed! Time: {:0>2}:{:0>2}:{:0>2}\n'.format(fold+1,h, m, s))

        cross_mse.append(best_mse)
        cross_mae.append(best_mae.item())
        cross_pcc.append(best_pcc.cpu().item())

    metrics_result = []

    validation_metrics = {
        'cross_mse': cross_mse,
        'cross_mae': cross_mae,
        'cross_pcc': cross_pcc
    }
    metrics_result.append(validation_metrics)

    with open(os.path.join(save_path, 'validation_metrics.txt'), 'w') as file:
        for key, values in validation_metrics.items():
            file.write(f"{key}: {values}\n")

    print('Cross Validation Results')
    print(f"MSE:{np.array(cross_mse).mean()} | MAE:{np.array(cross_mae).mean()} | PCC:{np.array(cross_pcc).mean()}")
