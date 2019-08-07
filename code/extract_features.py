import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import h5py

from config import cfg, cfg_from_file
from models.rgb_resnet import rgb_resnet50
from datasets import UCF101


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg')
    args = parser.parse_args()
    cfg_from_file(args.cfg)

    net = rgb_resnet50(pretrained=False, num_classes=101)
    params = torch.load(cfg.TRAIN.MODEL_PATH, map_location=device)
    net.load_state_dict(params['state_dict'])
    net = nn.Sequential(*list(net.children())[:-4])
    net.eval()
    net.to(device)

    dataset = UCF101(root=cfg.DATASET.DATA_DIR, split_file_path=cfg.DATASET.TRAIN_SPLIT_FILE_PATH)
    dataloader = DataLoader(dataset=dataset, batch_size=cfg.TRAIN.VAL_BATCH_SIZE, shuffle=True,
                                num_workers=cfg.WORKERS, drop_last=True)

    dataset_val = UCF101(root=cfg.DATASET.DATA_DIR, split_file_path=cfg.DATASET.VAL_SPLIT_FILE_PATH)
    dataloader_val = DataLoader(dataset=dataset_val, batch_size=cfg.TRAIN.VAL_BATCH_SIZE, drop_last=True,
                                    shuffle=False, num_workers=cfg.WORKERS)

    with torch.no_grad():
        with h5py.File(cfg.DATASET.FEATS_FILE_PATH, 'w') as hf:
            feats_list = []
            for image, answer in tqdm(dataloader):
                image = image.to(device)
                image = image.squeeze(1)
                feats = net(image)
                feats_list.append(feats.cpu().numpy())
            train_feats = np.concatenate(feats_list)
            hf.create_dataset('train', data=train_feats)

            feats_list = []
            for image, answer in tqdm(dataloader_val):
                image = image.to(device)
                image = image.squeeze(1)
                feats = net(image)
                feats_list.append(feats.cpu().numpy())
            val_feats = np.concatenate(feats_list)
            hf.create_dataset('val', data=val_feats)

        