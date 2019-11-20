from __future__ import print_function

from comet_ml import Experiment, ExistingExperiment

from pathlib import Path
import sys
import os
import shutil
from six.moves import range
import pprint
from tqdm import tqdm
from easydict import EasyDict as edict
import h5py
import numpy as np

from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision

from utils import mkdir_p, save_model, AverageMeter, \
    load_model, load_vocab, load_label_embeddings, get_labels_concepts_filename, calc_accuracy
from datasets.s2s_features import S2SFeatureDataset
from datasets.s2s_videos import VideoFolder
from datasets.transforms_video import *
from focal_loss import FocalLoss
import mac

from config import cfg


comet_args = {
    'project_name': 'mac-actions',
    'workspace': 'andresespinosapc',
}
if os.environ.get('COMET_DISABLE'):
    comet_args['disabled'] = True
    comet_args['api_key'] = ''
if cfg.TRAIN.RESUME_COMET_EXP_KEY:
    experiment = ExistingExperiment(previous_experiment=cfg.TRAIN.RESUME_COMET_EXP_KEY, **comet_args)
else:
    experiment = Experiment(**comet_args)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class Logger(object):
    def __init__(self, logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        sys.stdout.flush()
        self.log.flush()

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass

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
            sys.stdout = Logger(logfile=os.path.join(self.path, "logfile.log"))

        self.features_path = cfg.DATASET.FEATURES_PATH
        self.max_epochs = cfg.TRAIN.MAX_EPOCHS
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL

        # s_gpus = cfg.GPU_ID.split(',')
        # self.gpus = [int(ix) for ix in s_gpus]
        # self.num_gpus = len(self.gpus)

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
        # self.vocab = load_vocab(cfg)
        self.model, self.model_ema, self.concepts_per_label, self.mul_concepts_per_label = mac.load_model(cfg)
        self.weight_moving_average(alpha=0)
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

        if cfg.TRAIN.LOSS_FUNCTION == 'cross_entropy':
            self.main_loss_fn = torch.nn.CrossEntropyLoss().to(device)
        elif cfg.TRAIN.LOSS_FUNCTION == 'focal':
            alpha = cfg.TRAIN.FOCAL_LOSS_ALPHA
            gamma = cfg.TRAIN.FOCAL_LOSS_GAMMA
            self.main_loss_fn = FocalLoss(alpha, gamma).to(device)
        self.concept_loss_fn = torch.nn.BCELoss().to(device)

    # def loss_fn(self, target, scores, concepts_out):
    #     loss = self.main_loss_fn(scores, target)

    #     if self.cfg.MODEL.CONCEPT_AUX_TASK:
    #         concepts_target = self.concepts_per_label[target]
    #         loss += self.concept_loss_fn(concepts_out, concepts_target) * self.cfg.MODEL.CONCEPT_AUX_WEIGHT
        

    #     return loss

    # def loss_fn_multihead(self, target, scores_list):
    #     loss = 0

    #     concepts_target = self.mul_concepts_per_label[target]
    #     for i, scores in enumerate(scores_list):
    #         loss += self.main_loss_fn(scores, concepts_target[:, i])

    #     return loss

    def update_meters(self, loss, target, scores, preffix='', ema=False):
        if ema:
            preffix = 'ema_' + preffix
        batch_size = target.shape[0]
        self.meters[preffix + 'loss'].update(loss.item(), batch_size)
        if scores.shape[1] >= 5:
            top1, top5 = calc_accuracy(scores.detach().cpu(), target.detach().cpu(), topk=(1, 5))
            self.meters[preffix + 'top5'].update(top5, batch_size)
        else:
            top1, = calc_accuracy(scores.detach().cpu(), target.detach().cpu(), topk=(1,))
        self.meters[preffix + 'top1'].update(top1, batch_size)

    def calc_loss_and_update_meters(self, target, model_out, ema=False):
        batch_size = target.shape[0]
        loss = 0

        if self.cfg.MODEL.NAME == 'mac':
            scores, concepts_out = model_out
            loss += self.main_loss_fn(scores, target)
            if self.cfg.MODEL.CONCEPT_AUX_TASK:
                concepts_target = self.concepts_per_label[target]
                loss += self.concept_loss_fn(concepts_out, concepts_target) * self.cfg.MODEL.CONCEPT_AUX_WEIGHT

            self.update_meters(loss, target, scores, ema=ema)
        elif self.cfg.MODEL.NAME == 'i3d_multihead':
            concepts_target = self.mul_concepts_per_label[target]
            for i, scores in enumerate(model_out):
                cur_target = concepts_target[:, i]
                cur_loss = self.main_loss_fn(scores, cur_target)

                self.update_meters(cur_loss, cur_target, scores, preffix='head{}_'.format(i + 1), ema=ema)
                
                loss += cur_loss
        else:
            raise NotImplementedError('Model {} not implemented'.format(self.cfg.MODEL.NAME))

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

    def weight_moving_average(self, alpha=0.999):
        for param1, param2 in zip(self.model_ema.parameters(), self.model.parameters()):
            param1.data *= alpha
            param1.data += (1.0 - alpha) * param2.data

    def set_mode(self, mode="train"):
        if mode == "train":
            self.model.train()
            self.model_ema.train()
        else:
            self.model.eval()
            self.model_ema.eval()

    def reduce_lr(self):
        epoch_loss = self.total_epoch_loss / float(len(self.dataset) // self.batch_size)
        lossDiff = self.prior_epoch_loss - epoch_loss
        if ((lossDiff < 0.015 and self.prior_epoch_loss < 0.5 and self.lr > 0.00002) or \
            (lossDiff < 0.008 and self.prior_epoch_loss < 0.15 and self.lr > 0.00001) or \
            (lossDiff < 0.003 and self.prior_epoch_loss < 0.10 and self.lr > 0.000005)):
            self.lr *= 0.5
            print("Reduced learning rate to {}".format(self.lr))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr
        self.prior_epoch_loss = epoch_loss
        self.total_epoch_loss = 0

    def save_models(self, iteration, is_best=False):
        prefix = ""
        if is_best:
            prefix = "best_"
            # Remove previous best checkpoint
            for p in Path(self.model_dir).glob(prefix + '*'):
                p.unlink()
        save_model(self.model, self.optimizer, iteration, self.model_dir, model_name=prefix+"model")
        save_model(self.model_ema, None, iteration, self.model_dir, model_name=prefix+"model_ema")

    def load_models(self, model_dir, iteration):
        load_model(self.model, self.optimizer, iteration, model_dir, model_name='model')
        load_model(self.model_ema, None, iteration, model_dir, model_name='model_ema')

    def init_meters(self, ema=False):
        if self.cfg.MODEL.NAME == 'mac':
            self.meters = {
                'loss': AverageMeter(),
                'top1': AverageMeter(),
                'top5': AverageMeter(),
            }
        elif self.cfg.MODEL.NAME == 'i3d_multihead':
            n_heads = self.model.n_heads
            self.meters = {}
            for i in range(n_heads):
                preffix = 'head{}_'.format(i + 1)
                self.meters.update({
                    preffix + 'loss': AverageMeter(),
                    preffix + 'top1': AverageMeter(),
                    preffix + 'top5': AverageMeter(),
                })
        else:
            raise NotImplementedError('Model {} not implemented'.format(self.cfg.MODEL.NAME))

        if ema:
            for key in list(self.meters.keys()):
                self.meters['ema_' + key] = AverageMeter()

    # def log_progress(self, model_out, target):
    #     if self.cfg.MODEL.NAME == 'mac':
    #         top1, top5 = calc_accuracy(model_out.detach().cpu(), target.detach().cpu(), topk=(1, 5))
    #         self.meters['loss'].update(loss.item(), target.shape[0])
    #         self.meters['top1'].update(top1, target.shape[0])
    #         self.meters['top5'].update(top5, target.shape[0])
    #     elif self.cfg.MODEL.NAME == 'i3d_multihead':
    #         concepts_target = self.mul_concepts_per_label[target]
    #         for i, scores in enumerate(model_out):
    #             preffix = 'head{}_'.format(i + 1)
    #             self.meters[preffix + 'loss'].update(loss.item(), target.shape[0])
    #             if scores.shape[1] >= 5:
    #                 top1, top5 = calc_accuracy(scores.detach().cpu(), concepts_target[:, i].detach().cpu(), topk=(1, 5))
    #                 self.meters[preffix + 'top5'].update(top5, target.shape[0])
    #             else:
    #                 top1, = calc_accuracy(scores.detach().cpu(), concepts_target[:, i].detach().cpu(), topk=(1,))
                
    #             self.meters[preffix + 'top1'].update(top1, target.shape[0])
                

    def train_epoch(self, epoch):
        cfg = self.cfg
        avg_loss = 0
        train_accuracy = 0

        self.labeled_data = iter(self.dataloader)
        self.set_mode("train")

        pbar = tqdm(self.labeled_data)
        self.optimizer.zero_grad()

        self.init_meters()
        for i, (image, target) in enumerate(pbar):
            ######################################################
            # (1) Prepare training data
            ######################################################
            image = image.to(device)
            target = target.long().to(device)

            ############################
            # (2) Train Model
            ############################
            model_out = self.model(image)
            loss = self.calc_loss_and_update_meters(target, model_out)

            loss /= self.iter_to_step

            loss.backward()
            if (i+1) % self.iter_to_step == 0:
                if self.cfg.TRAIN.CLIP_GRADS:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.TRAIN.CLIP)

                self.optimizer.step()
                self.weight_moving_average()

                self.optimizer.zero_grad()

            # accuracy = top1
            # if avg_loss == 0:
            #     avg_loss = loss.item()
            #     train_accuracy = accuracy
            # else:
            #     avg_loss = 0.99 * avg_loss + 0.01 * loss.item()
            #     train_accuracy = 0.99 * train_accuracy + 0.01 * accuracy
            # self.total_epoch_loss += loss.item()

            if self.cfg.MODEL.NAME == 'mac':
                pbar.set_description(
                    'Epoch: {}; Avg Loss: {:.5f}; Avg Top1: {:.5f}; Avg Top5: {:.5f}'.format(
                        epoch + 1,
                        self.meters['loss'].avg,
                        self.meters['top1'].avg,
                        self.meters['top5'].avg,
                    )
                )
            elif self.cfg.MODEL.NAME == 'i3d_multihead':
                pbar.set_description(
                    'Epoch: {}; Head1 Top1: {:.5f}; Head2 Top1: {:.5f}; Head3 Top1: {:.5f}; Head4 Top1: {:.5f}'.format(
                        epoch + 1,
                        self.meters['head1_top1'].avg,
                        self.meters['head2_top1'].avg,
                        self.meters['head3_top1'].avg,
                        self.meters['head4_top1'].avg,
                    )
                )
            else:
                raise NotImplementedError('Model {} not implemented'.format(self.cfg.MODEL.NAME))

        metrics = { 'avg_{}'.format(key): meter.avg for key, meter in self.meters.items() }
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

            # Evaluate multihead
            # if cfg.MODEL.NAME == 'i3d_multihead':
            #     with experiment.test():
            #         for i in range(5):
            #             self.train_epoch()
            #             self.log_results()

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

        if self.cfg.MODEL.NAME == 'mac':
            val_accuracy, val_accuracy_ema = val_metrics['avg_top1'], val_metrics['avg_top1_ema']
            print("Epoch: {}\tVal Top1: {},\tVal Top1 EMA: {},\tVal Avg Loss: {},\tLR: {}".
                format(epoch, val_accuracy, val_accuracy_ema, val_metrics['avg_loss'], self.lr))
        elif self.cfg.MODEL.NAME == 'i3d_multihead':
            val_accuracy, val_accuracy_ema = val_metrics['avg_head1_top1'], val_metrics['avg_ema_head1_top1']
            print("Epoch: {}\tVal Head1Top1 EMA: {},\tVal Head2Top1 EMA: {},\tVal Head3Top1 EMA: {},\tVal Head4Top1 EMA: {},\tLR: {}".
                format(epoch, val_metrics['avg_ema_head1_top1'], val_metrics['avg_ema_head2_top1'], val_metrics['avg_ema_head3_top1'], val_metrics['avg_ema_head4_top1'], self.lr))
        else:
            raise NotImplementedError('Model {} not implemented'.format(self.cfg.MODEL.NAME))

        if val_accuracy > self.previous_best_acc:
            self.previous_best_acc = val_accuracy
            self.previous_best_epoch = epoch
            self.save_models(epoch, is_best=True)

        if epoch % self.snapshot_interval == 0:
            self.save_models(epoch)

    def calc_metrics(self, mode="train", max_samples=None):
        self.set_mode("validation")

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

        self.init_meters(ema=True)

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
                model_out = self.model(image)
                ema_model_out = self.model_ema(image)
                loss = self.calc_loss_and_update_meters(target, model_out)
                loss_ema = self.calc_loss_and_update_meters(target, ema_model_out)

        # TEMP
        if self.cfg.MODEL.NAME != 'i3d_multihead':
            self.scheduler.step(self.meters['loss'].avg)

        metrics = { 'avg_{}'.format(key): meter.avg for key, meter in self.meters.items() }
        experiment.log_metrics(metrics)

        return metrics
