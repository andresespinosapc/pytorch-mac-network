from __future__ import print_function

from comet_ml import Experiment

import sys
import os
import shutil
from six.moves import range
import pprint
from tqdm import tqdm
from easydict import EasyDict as edict
import h5py

from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from utils import mkdir_p, save_model, load_model, load_vocab, load_label_embeddings
from datasets import S2SFeatureDataset, collate_fn
import mac


experiment = Experiment(project_name='mac-actions', workspace='andresespinosapc', disable=True, api_key='')

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

        s_gpus = cfg.GPU_ID.split(',')
        self.gpus = [int(ix) for ix in s_gpus]
        self.num_gpus = len(self.gpus)

        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.val_batch_size = self.cfg.TRAIN.VAL_BATCH_SIZE
        self.lr = cfg.TRAIN.LEARNING_RATE

        if cfg.CUDA:
            torch.cuda.set_device(self.gpus[0])
            cudnn.benchmark = True

        # load dataset
        self.dataset = S2SFeatureDataset(self.features_path, split='train')
        # self.dataset = Subset(self.dataset, range(5))
        self.dataloader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=True,
                                       num_workers=cfg.WORKERS, drop_last=False)

        self.dataset_val = S2SFeatureDataset(self.features_path, split='val')
        # self.dataset_val = Subset(self.dataset_val, range(5))
        self.dataloader_val = DataLoader(dataset=self.dataset_val, batch_size=self.val_batch_size, drop_last=False,
                                         shuffle=False, num_workers=cfg.WORKERS)

        # load model
        with h5py.File(cfg.DATASET.LABELS_CONCEPTS_PATH) as h5f:
            self.labels_matrix = h5f['labels_matrix'][:]
            self.concepts = h5f['concepts'][:]
        # self.vocab = load_vocab(cfg)
        # TEMP
        self.vocab = { 'question_token_to_idx': [] }
        self.model, self.model_ema = mac.load_MAC(cfg, self.vocab, self.labels_matrix, self.concepts)
        self.weight_moving_average(alpha=0)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        # self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        #     self.optimizer, mode='min', factor=.1, patience=5,
        # )
        if cfg.TRAIN.RESUME_SNAPSHOT_DIR != '':
            model_dir = os.path.join('data', cfg.TRAIN.RESUME_SNAPSHOT_DIR, 'Model')
            self.load_models(model_dir, cfg.TRAIN.RESUME_SNAPSHOT_ITER)

        self.previous_best_acc = 0.0
        self.previous_best_epoch = 0

        self.total_epoch_loss = 0
        self.prior_epoch_loss = 10

        self.print_info()
        self.loss_fn = torch.nn.CrossEntropyLoss().to(device)

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
        save_model(self.model, self.optimizer, iteration, self.model_dir, model_name=prefix+"model")
        save_model(self.model_ema, None, iteration, self.model_dir, model_name=prefix+"model_ema")

    def load_models(self, model_dir, iteration):
        load_model(self.model, self.optimizer, iteration, model_dir, model_name='model')
        load_model(self.model_ema, None, iteration, model_dir, model_name='model_ema')

    def train_epoch(self, epoch):
        cfg = self.cfg
        avg_loss = 0
        train_accuracy = 0

        self.labeled_data = iter(self.dataloader)
        self.set_mode("train")

        dataset = tqdm(self.labeled_data)

        for image, target in dataset:
            ######################################################
            # (1) Prepare training data
            ######################################################
            image = image.to(device)
            target = target.long().to(device)

            ############################
            # (2) Train Model
            ############################
            self.optimizer.zero_grad()

            scores = self.model(image)
            loss = self.loss_fn(scores, target)
            loss.backward()

            if self.cfg.TRAIN.CLIP_GRADS:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.TRAIN.CLIP)

            self.optimizer.step()
            self.weight_moving_average()

            ############################
            # (3) Log Progress
            ############################
            correct = scores.detach().argmax(1) == target
            accuracy = correct.sum().cpu().numpy() / target.shape[0]
            metrics = {
                'loss': loss.item(),
                'accuracy': accuracy,
            }
            experiment.log_metrics(metrics)

            if avg_loss == 0:
                avg_loss = loss.item()
                train_accuracy = accuracy
            else:
                avg_loss = 0.99 * avg_loss + 0.01 * loss.item()
                train_accuracy = 0.99 * train_accuracy + 0.01 * accuracy
            self.total_epoch_loss += loss.item()

            dataset.set_description(
                'Epoch: {}; Avg Loss: {:.5f}; Avg Train Acc: {:.5f}'.format(epoch + 1, avg_loss, train_accuracy)
            )

        metrics = {
            'avg_loss': avg_loss,
            'epoch': epoch,
        }
        experiment.log_metrics(metrics)
        dict = {
            "avg_loss": avg_loss,
            "train_accuracy": train_accuracy
        }

        return dict

    def train(self):
        cfg = self.cfg
        print("Start Training")
        for epoch in range(self.max_epochs):
            with experiment.train():
                dict = self.train_epoch(epoch)
            # self.scheduler.step(dict['avg_loss'])
            with experiment.validate():
                self.log_results(epoch, dict)
            if cfg.TRAIN.EALRY_STOPPING:
                if epoch - cfg.TRAIN.PATIENCE == self.previous_best_epoch:
                    break

        self.save_models(self.max_epochs)
        self.writer.close()
        print("Finished Training")
        print("Highest validation accuracy: {} at epoch {}")

    def log_results(self, epoch, dict, max_eval_samples=None):
        epoch += 1
        self.writer.add_scalar("avg_loss", dict["avg_loss"], epoch)
        self.writer.add_scalar("train_accuracy", dict["train_accuracy"], epoch)

        val_accuracy, val_accuracy_ema = self.calc_accuracy("validation", max_samples=max_eval_samples)
        self.writer.add_scalar("val_accuracy_ema", val_accuracy_ema, epoch)
        self.writer.add_scalar("val_accuracy", val_accuracy, epoch)

        print("Epoch: {}\tVal Acc: {},\tVal Acc EMA: {},\tAvg Loss: {},\tLR: {}".
              format(epoch, val_accuracy, val_accuracy_ema, dict["avg_loss"], self.lr))

        if val_accuracy > self.previous_best_acc:
            self.previous_best_acc = val_accuracy
            self.previous_best_epoch = epoch
            self.save_models(epoch, is_best=True)

        if epoch % self.snapshot_interval == 0:
            self.save_models(epoch)

    def calc_accuracy(self, mode="train", max_samples=None):
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

        all_accuracies = []
        all_accuracies_ema = []

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
                scores = self.model(image)
                scores_ema = self.model_ema(image)

            correct_ema = scores_ema.detach().argmax(1) == target
            accuracy_ema = correct_ema.sum().cpu().numpy() / target.shape[0]
            all_accuracies_ema.append(accuracy_ema)

            correct = scores.detach().argmax(1) == target
            accuracy = correct.sum().cpu().numpy() / target.shape[0]
            all_accuracies.append(accuracy)

        accuracy_ema = sum(all_accuracies_ema) / float(len(all_accuracies_ema))
        accuracy = sum(all_accuracies) / float(len(all_accuracies))
        experiment.log_metric('accuracy', accuracy)

        return accuracy, accuracy_ema
