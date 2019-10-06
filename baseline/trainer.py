from __future__ import print_function

import sys
import os
import shutil
from six.moves import range
import pprint
from tqdm import tqdm

from comet_ml import Experiment

from pathlib import Path
from easydict import EasyDict as edict
import h5py
import numpy as np

from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision

from utils import mkdir_p, save_model, AverageMeter, \
    load_model, load_vocab, load_label_embeddings, get_labels_concepts_filename, calc_accuracy

from datasets.s2s_features import S2SFeatureDataset
from datasets.s2s_videos import VideoFolder
from datasets.transforms_video import *
from i3d import I3D, Mixed, Unit3Dpy

import argparse
import random
import datetime
import dateutil
import dateutil.tz

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

from config import cfg, cfg_from_file

comet_args = {
    'project_name': 'mac-actions',
    'workspace': 'andresespinosapc',
}
if os.environ.get('COMET_DISABLE'):
    comet_args['disabled'] = True
    comet_args['api_key'] = ''
experiment = Experiment(**comet_args)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def log_comet_parameters(cfg):
    for key in cfg.keys():
        if type(cfg[key]) is not edict:
            experiment.log_parameter(key, cfg[key])
        else:
            for key2 in cfg[key].keys():
                experiment.log_parameter('{}_{}'.format(key, key2), cfg[key][key2])

class Trainer():
    def __init__(self, log_dir, cfg):
        log_comet_parameters(cfg)

        self.path = log_dir
        self.cfg = cfg

        if cfg.TRAIN.FLAG:
            self.model_dir = os.path.join(self.path, 'Model')
            self.log_dir = os.path.join(self.path, 'Log')
            mkdir_p(self.model_dir)
            mkdir_p(self.log_dir)
            self.writer = SummaryWriter(log_dir=self.log_dir)

        self.features_path = cfg.DATASET.FEATURES_PATH
        self.max_epochs = cfg.TRAIN.MAX_EPOCHS
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL

        self.batch_size = cfg.TRAIN.REAL_BATCH_SIZE
        effective_batch_size = cfg.TRAIN.EFFECTIVE_BATCH_SIZE
        if effective_batch_size % self.batch_size != 0:
            raise ValueError('Effective batch size must be a multiple of real batch size')
        self.iter_to_step = int(effective_batch_size / self.batch_size)
        self.val_batch_size = self.cfg.TRAIN.VAL_BATCH_SIZE
        self.lr = cfg.TRAIN.LEARNING_RATE

        if cfg.CUDA:
            # torch.cuda.set_device(self.gpus[0])
            cudnn.benchmark = True

        # load dataset
        if cfg.DATASET.DATA_TYPE == 'features':
            self.dataset = S2SFeatureDataset(self.features_path, split='train')
            self.dataset_val = S2SFeatureDataset(self.features_path, split='val')
        elif cfg.DATASET.DATA_TYPE == 'videos':
            # define augmentation pipeline
            upscale_size_train = int(cfg.DATASET.INPUT_SPATIAL_SIZE * cfg.DATASET.UPSCALE_FACTOR_TRAIN)
            upscale_size_eval = int(cfg.DATASET.INPUT_SPATIAL_SIZE * cfg.DATASET.UPSCALE_FACTOR_EVAL)
            # Random crop videos during training
            transform_train_pre = ComposeMix([
                    [RandomRotationVideo(15), "vid"],
                    [Scale(upscale_size_train), "img"],
                    [RandomCropVideo(cfg.DATASET.INPUT_SPATIAL_SIZE), "vid"],
                    ])

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
            self.dataset = VideoFolder(
                root=cfg.DATASET.DATA_FOLDER,
                json_file_input=cfg.DATASET.TRAIN_JSON_PATH,
                json_file_labels=cfg.DATASET.LABELS_JSON_PATH,
                clip_size=cfg.DATASET.CLIP_SIZE,
                nclips=cfg.DATASET.NCLIPS_TRAIN,
                step_size=cfg.DATASET.STEP_SIZE_TRAIN,
                is_val=False,
                transform_pre=transform_train_pre,
                transform_post=transform_post,
            )
            self.dataset_val = VideoFolder(
                root=cfg.DATASET.DATA_FOLDER,
                json_file_input=cfg.DATASET.VAL_JSON_PATH,
                json_file_labels=cfg.DATASET.LABELS_JSON_PATH,
                clip_size=cfg.DATASET.CLIP_SIZE,
                nclips=cfg.DATASET.NCLIPS_VAL,
                step_size=cfg.DATASET.STEP_SIZE_VAL,
                is_val=True,
                transform_pre=transform_eval_pre,
                transform_post=transform_post,
            )
        else:
            raise NotImplementedError('Invalid dataset data_type from config')
        # self.dataset = Subset(self.dataset, range(5))
        # self.dataset_val = Subset(self.dataset_val, range(5))
        self.dataloader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=True,
                                       num_workers=cfg.WORKERS, drop_last=False)
        self.dataloader_val = DataLoader(dataset=self.dataset_val, batch_size=self.val_batch_size, drop_last=False,
                                         shuffle=False, num_workers=cfg.WORKERS)

        # load model
        self.model = I3D(num_classes=400, modality='rgb')
        self.model.load_state_dict(torch.load('kinetics_i3d_pytorch/model/model_rgb.pth'))
        self.prepare_to_train()
        self.model.to(device)

        if cfg.TRAIN.OPTIMIZER == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        elif cfg.TRAIN.OPTIMIZER == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.lr,
                momentum=cfg.TRAIN.MOMENTUM,
                weight_decay=cfg.TRAIN.WEIGHT_DECAY,
            )
        else:
            raise NotImplementedError('Invalid train optimizer')

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=.1, patience=5,
        )
        if cfg.TRAIN.RESUME_SNAPSHOT_DIR != '':
            model_dir = os.path.join('data', cfg.TRAIN.RESUME_SNAPSHOT_DIR, 'Model')
            self.load_models(model_dir, cfg.TRAIN.RESUME_SNAPSHOT_ITER)

        self.previous_best_acc = 0.0
        self.previous_best_epoch = 0

        self.total_epoch_loss = 0
        self.prior_epoch_loss = 10

        self.print_info()

        self.main_loss_fn = torch.nn.CrossEntropyLoss().to(device)

    def loss_fn(self, target, scores):
        loss = self.main_loss_fn(scores, target)
        return loss

    def print_info(self):
        print('Using config:')
        pprint.pprint(self.cfg)
        print("\n")

        pprint.pprint("Size of dataset: {}".format(len(self.dataset)))
        print("\n")

        print("Using MAC-Model:")
        pprint.pprint(self.model)
        print("\n")

    def prepare_to_train(self):
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.mixed_5b = Mixed(832, [256, 160, 320, 32, 128, 128])
        self.model.mixed_5c = Mixed(832, [384, 192, 384, 48, 128, 128])
        self.model.avg_pool = torch.nn.AvgPool3d((2, 7, 7), (1, 1, 1))

        self.model.conv3d_0c_1x1 = Unit3Dpy(
                in_channels=1024,
                out_channels=18,
                kernel_size=(1, 1, 1),
                activation=None,
                use_bias=True,
                use_bn=False)

        #TODO

    def save_models(self, iteration, is_best=False):
        prefix = ""
        if is_best:
            prefix = "best_"
            # Remove previous best checkpoint
            for p in Path(self.model_dir).glob(prefix + '*'):
                p.unlink()
        save_model(self.model, self.optimizer, iteration, self.model_dir, model_name=prefix+"model")

    def load_models(self, model_dir, iteration):
        load_model(self.model, self.optimizer, iteration, model_dir, model_name='model')

    def train_epoch(self, epoch):
        loss_meter = AverageMeter()
        top1_meter = AverageMeter()
        top5_meter = AverageMeter()

        cfg = self.cfg
        avg_loss = 0
        train_accuracy = 0

        self.labeled_data = iter(self.dataloader)
        self.model.train()

        pbar = tqdm(self.labeled_data)
        self.optimizer.zero_grad()

        for i, (image, target) in enumerate(pbar):
            ######################################################
            # (1) Prepare training data
            ######################################################
            image = image.to(device)
            target = target.long().to(device)

            ############################
            # (2) Train Model
            ############################
            _, scores = self.model(image)
            loss = self.loss_fn(target, scores)

            loss /= self.iter_to_step

            loss.backward()
            if (i+1) % self.iter_to_step == 0:
                if self.cfg.TRAIN.CLIP_GRADS:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.TRAIN.CLIP)

                self.optimizer.step()
                self.optimizer.zero_grad()

            ############################
            # (3) Log Progress
            ############################
            top1, top5 = calc_accuracy(scores.detach().cpu(), target.detach().cpu(), topk=(1, 5))
            loss_meter.update(loss.item(), target.shape[0])
            top1_meter.update(top1, target.shape[0])
            top5_meter.update(top5, target.shape[0])

            pbar.set_description(
                'Epoch: {}; Avg Loss: {:.5f}; Avg Top1: {:.5f}; Avg Top5: {:.5f}'.format(
                    epoch + 1, loss_meter.avg, top1_meter.avg, top5_meter.avg
                )
            )

        metrics = {
            'avg_loss': loss_meter.avg,
            'avg_top1': top1_meter.avg,
            'avg_top5': top5_meter.avg,
        }
        experiment.log_metrics(metrics)

        return metrics

    def train(self):
        cfg = self.cfg
        print("Start Training")
        for epoch in range(self.max_epochs):
            with experiment.train():
                metrics = self.train_epoch(epoch)
            # self.scheduler.step(metrics['avg_loss'])
            with experiment.validate():
                self.log_results(epoch, metrics)
            experiment.log_metric('epoch', epoch)
            if cfg.TRAIN.EARLY_STOPPING:
                if epoch - cfg.TRAIN.PATIENCE == self.previous_best_epoch:
                    break

        self.save_models(self.max_epochs)
        self.writer.close()
        print("Finished Training")
        print("Highest validation accuracy: {} at epoch {}")

    def log_results(self, epoch, train_metrics, max_eval_samples=None):
        epoch += 1
        for k, v in train_metrics.items():
            self.writer.add_scalar('train_{}'.format(k), v, epoch)

        val_metrics = self.calc_metrics("validation", max_samples=max_eval_samples)
        for k, v in val_metrics.items():
            self.writer.add_scalar('val_{}'.format(k), v, epoch)

        val_accuracy = val_metrics['avg_top1']
        print("Epoch: {}\tVal Top1: {},\tVal Avg Loss: {},\tLR: {}".
              format(epoch, val_accuracy, val_metrics['avg_loss'], self.lr))

        if val_accuracy > self.previous_best_acc:
            self.previous_best_acc = val_accuracy
            self.previous_best_epoch = epoch
            self.save_models(epoch, is_best=True)

        if epoch % self.snapshot_interval == 0:
            self.save_models(epoch)

    def calc_metrics(self, mode="train", max_samples=None):
        self.model.eval()

        if mode == "train":
            eval_data = iter(self.dataloader)
            num_imgs = len(self.dataset)
        elif mode == "validation":
            eval_data = iter(self.dataloader_val)
            num_imgs = len(self.dataset_val)

        batch_size = self.val_batch_size
        total_iters = num_imgs // batch_size
        if max_samples is not None:
            max_iter = max_samples // batch_size
        else:
            max_iter = None

        loss_meter = AverageMeter()
        top1_meter = AverageMeter()
        top5_meter = AverageMeter()

        for _iteration in range(total_iters):
            try:
                data = next(eval_data)
            except StopIteration:
                break
            if max_iter is not None and _iteration == max_iter:
                break

            image, target = data
            image = image.to(device)
            target = target.long().to(device)

            with torch.no_grad():
                _, scores = self.model(image)
                loss = self.loss_fn(target, scores)
                loss_meter.update(loss.item(), target.shape[0])

            top1, top5 = calc_accuracy(scores.detach().cpu(), target.detach().cpu(), topk=(1, 5))
            top1_meter.update(top1, target.shape[0])
            top5_meter.update(top5, target.shape[0])

        self.scheduler.step(loss_meter.avg)
        metrics = {
            'avg_loss': loss_meter.avg,
            'avg_top1': top1_meter.avg,
            'avg_top5': top5_meter.avg,
        }
        experiment.log_metrics(metrics)

        return metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_file', help='optional config file', default='shapes_train.yml', type=str)
    parser.add_argument('--gpu',  dest='gpu_id', type=str, default='0')
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()
    return args


def set_logdir(max_steps):
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    logdir = "data/{}_max_steps_{}".format(now, max_steps)
    mkdir_p(logdir)
    print("Saving output to: {}".format(logdir))
    code_dir = os.path.join(os.getcwd(), "code")
    mkdir_p(os.path.join(logdir, "Code"))
    for filename in os.listdir(code_dir):
        if filename.endswith(".py"):
            shutil.copy(code_dir + "/" + filename, os.path.join(logdir, "Code"))
    shutil.copy(args.cfg_file, logdir)
    return logdir


if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.gpu_id != -1:
        cfg.GPU_ID = args.gpu_id
    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)

    if cfg.TRAIN.FLAG:
        logdir = set_logdir(cfg.TRAIN.MAX_STEPS)
        trainer = Trainer(logdir, cfg)
        trainer.train()
    else:
        raise NotImplementedError
