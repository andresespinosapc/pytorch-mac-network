import os
import cv2
import sys
import importlib
import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import numpy as np

# sys.path.insert(0, "../")

# imports for displaying a video an IPython cell
import io
import base64

from datasets.transforms_video import *
from datasets.s2s_videos import VideoFolder

from config import cfg, cfg_from_file

from i3d import I3DFeats, Mixed, Unit3Dpy

from pprint import pprint

from tqdm import tqdm
import h5py
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--config_file', default='cfg/train_s2s_mac.yml')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--out_file', default='feats.h5')
parser.add_argument('--train', action='store_true')
parser.add_argument('--val', action='store_true')
parser.add_argument('--test', action='store_true')
args = parser.parse_args()

base_dir = '.'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('Using device:', device)

# Load config file
cfg_from_file(args.config_file)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = I3DFeats(num_classes=400, modality='rgb').to(device)
model.load_state_dict(torch.load('kinetics_i3d_pytorch/model/model_rgb.pth'))
model.eval()

upscale_size_eval = int(cfg.DATASET.INPUT_SPATIAL_SIZE * cfg.DATASET.UPSCALE_FACTOR_EVAL)

# Center crop videos during evaluation
transform_eval_pre = ComposeMix([
        [Scale(upscale_size_eval), "img"],
        [torchvision.transforms.ToPILImage(), "img"],
        [torchvision.transforms.CenterCrop(cfg.DATASET.INPUT_SPATIAL_SIZE), "img"],
        ])

# Transforms common to train and eval sets and applied after "pre" transforms
transform_post = ComposeMix([
        [torchvision.transforms.ToTensor(), "img"],
        [torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # default values for imagenet
                std=[0.229, 0.224, 0.225]), "img"]
        ])

dataset = VideoFolder(
    root=cfg.DATASET.DATA_FOLDER,
    json_file_input=cfg.DATASET.TRAIN_JSON_PATH,
    json_file_labels=cfg.DATASET.LABELS_JSON_PATH,
    clip_size=cfg.DATASET.CLIP_SIZE,
    nclips=cfg.DATASET.NCLIPS_TRAIN,
    step_size=cfg.DATASET.STEP_SIZE_TRAIN,
    is_val=True,
    transform_pre=transform_eval_pre,
    transform_post=transform_post,
    get_item_id=True,
)
dataset_val = VideoFolder(
    root=cfg.DATASET.DATA_FOLDER,
    json_file_input=cfg.DATASET.VAL_JSON_PATH,
    json_file_labels=cfg.DATASET.LABELS_JSON_PATH,
    clip_size=cfg.DATASET.CLIP_SIZE,
    nclips=cfg.DATASET.NCLIPS_VAL,
    step_size=cfg.DATASET.STEP_SIZE_VAL,
    is_val=True,
    transform_pre=transform_eval_pre,
    transform_post=transform_post,
    get_item_id=True,
)

# dataset = Subset(dataset, range(10))
# dataset_val = Subset(dataset_val, range(10))

train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
val_dataloader = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
# test_data = VideoFolder(root=config['data_folder'],
#                        json_file_input=config['json_data_test'],
#                        json_file_labels=config['json_file_labels'],
#                        clip_size=config['clip_size'],
#                        nclips=config['nclips_val'],
#                        step_size=config['step_size_val'],
#                        is_val=True,
#                        transform_pre=transform_eval_pre,
#                        transform_post=transform_post,
#                        get_item_id=True,
#                        )
# test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

def save_features(dataloader, h5f, out_shape, split):
    data_len = len(dataloader.dataset)
    batch_size = dataloader.batch_size
    group = h5f.create_group(split)
    data = group.create_dataset('data', (data_len, *out_shape), dtype='f')
    targets = group.create_dataset('target', (data_len,), dtype='uint8')
    video_ids = group.create_dataset('video_id', (data_len,), dtype='int32')
    with torch.no_grad():
        pbar = tqdm(dataloader)
        for i, (input_data, target, item_id) in enumerate(pbar):
            input_data = input_data.to(device)
            out = model(input_data)
            start = i * batch_size
            end = i * batch_size + out.shape[0]
            data[start:end] = out.detach().cpu().numpy()
            targets[start:end] = target
            video_ids[start:end] = list(map(int, item_id))

# TEMP
out_shape = (832, 18, 14, 14)
with h5py.File(args.out_file, 'w') as h5f:
    if args.train:
        save_features(train_dataloader, h5f, out_shape, 'train')
    if args.val:
        save_features(val_dataloader, h5f, out_shape, 'val')
    if args.test:
        raise NotImplementedError('Test feature extraction')
        save_features(test_dataloader, h5f, out_shape, 'test')
