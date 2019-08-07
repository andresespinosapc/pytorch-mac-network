from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import cv2
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import PIL
import os
import os.path
import pickle
import random
import numpy as np
import glob
from pathlib import Path
import json
import h5py
import re
import random

from config import cfg
import video_transforms


class UCF101(data.Dataset):
    def __init__(self, root, split_file_path, n_frames=1, transform=None):
        if n_frames > 1:
            raise NotImplementedError('Number of frames > 1')

        self.transform = transform
        if transform is None:
            # self.transform = transforms.Compose([
            #     transforms.Resize(256),
            #     transforms.CenterCrop(224),
            #     transforms.ToTensor(),
            #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            # ])
            is_color = True
            scale_ratios = [1.0, 0.875, 0.75, 0.66]
            clip_mean = [0.485, 0.456, 0.406]
            clip_std = [0.229, 0.224, 0.225]
            normalize = video_transforms.Normalize(mean=clip_mean, std=clip_std)
            if 'train' in split_file_path:
                self.transform = video_transforms.Compose([
                    # video_transforms.Scale((256)),
                    video_transforms.MultiScaleCrop((224, 224), scale_ratios),
                    video_transforms.RandomHorizontalFlip(),
                    video_transforms.ToTensor(),
                    normalize,
                ])
            else:
                self.transform = video_transforms.Compose([
                    # video_transforms.Scale((256)),
                    video_transforms.CenterCrop((224)),
                    video_transforms.ToTensor(),
                    normalize,
                ])
        self.root = root
        data = []
        for line in open(split_file_path):
            file_name, duration_str, target_str = line.split(' ')
            duration, target = int(duration_str), int(target_str)
            # TEMP
            if 'HandStand' in file_name:
                file_name = file_name.replace('HandStand', 'Handstand')
            frames_path = os.path.join(self.root, file_name)
            # Take the frame in the middle
            frames_ids = np.array([int(duration / 2)], dtype=np.int)
            # diff = (duration - 1) / (n_frames - 1)
            # frames_ids = (np.arange(n_frames) * diff + 1).astype(np.int)
            new_data = [frames_path, frames_ids, target]
            data.append(new_data)
        self.data = np.array(data)
    
    def get_transformed_frame(self, path):
        # img = Image.open(path).convert('RGB')
        # return self.transform(img)

        cv_read_flag = cv2.IMREAD_COLOR
        interpolation = cv2.INTER_LINEAR
        cv_img_origin = cv2.imread(path, cv_read_flag)
        new_width = 340
        new_height = 256
        cv_img = cv2.resize(cv_img_origin, (new_width, new_height), interpolation)
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

        return self.transform(cv_img)

    def __getitem__(self, index):
        frames_path, frames_ids, target = self.data[index]
        frames = []
        for i in range(len(frames_ids)):
            frame_path = os.path.join(frames_path, 'frame{}.jpg'.format(str(frames_ids[i]).zfill(6)))
            sys.stdout.flush()
            img = self.get_transformed_frame(frame_path)
            frames.append(img)
        return torch.stack(frames), target

    def __len__(self):
        return self.data.shape[0]

class UCF101Feats(data.Dataset):
    def __init__(self, feats_file_path, split_file_path):
        with h5py.File(feats_file_path, 'r') as hf:
            if 'train' in split_file_path:
                self.feats = hf['train'][:]
            else:
                self.feats = hf['val'][:]

        targets = []
        for i, line in enumerate(open(split_file_path)):
            file_name, duration_str, target_str = line.split(' ')
            duration, target = int(duration_str), int(target_str)
            targets.append(target)
        self.targets = np.array(targets)

    def __getitem__(self, index):
        return self.feats[index], self.targets[index]

    def __len__(self):
        return self.feats.shape[0]    


class ClevrDataset(data.Dataset):
    def __init__(self, data_dir, split='train'):

        with open(os.path.join(data_dir, '{}.pkl'.format(split)), 'rb') as f:
            self.data = pickle.load(f)
        self.img = h5py.File(os.path.join(data_dir, '{}_features.h5'.format(split)), 'r')['features']

    def __getitem__(self, index):
        imgfile, question, answer, family = self.data[index]
        id = int(imgfile.rsplit('_', 1)[1][:-4])
        img = torch.from_numpy(self.img[id])

        return img, question, len(question), answer, family

    def __len__(self):
        return len(self.data)


def collate_fn(batch):
    images, lengths, answers, _ = [], [], [], []
    batch_size = len(batch)

    max_len = max(map(lambda x: len(x[1]), batch))

    questions = np.zeros((batch_size, max_len), dtype=np.int64)
    sort_by_len = sorted(batch, key=lambda x: len(x[1]), reverse=True)

    for i, b in enumerate(sort_by_len):
        image, question, length, answer, family = b
        images.append(image)
        length = len(question)
        questions[i, :length] = question
        lengths.append(length)
        answers.append(answer)

    return {'image': torch.stack(images), 'question': torch.from_numpy(questions),
            'answer': torch.LongTensor(answers), 'question_length': lengths}

def collate_fn_wo_questions(batch):
    return batch

# def collate_fn_wo_questions(batch):
#     images, lengths, answers, _ = [], [], [], []
#     batch_size = len(batch)

#     max_len = max(map(lambda x: len(x[1]), batch))

#     questions = np.zeros((batch_size, max_len), dtype=np.int64)
#     sort_by_len = sorted(batch, key=lambda x: len(x[1]), reverse=True)

#     for i, b in enumerate(sort_by_len):
#         image, question, length, answer, family = b
#         images.append(image)
#         length = len(question)
#         questions[i, :length] = question
#         lengths.append(length)
#         answers.append(answer)

#     return {'image': torch.stack(images), 'question': torch.from_numpy(questions),
#             'answer': torch.LongTensor(answers), 'question_length': lengths}
