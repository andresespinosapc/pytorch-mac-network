import os
import errno
import numpy as np
import glob
import pickle
import json

from copy import deepcopy
from config import cfg

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from torch.nn import init
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import functional as F
import torchvision.utils as vutils


# Taken from PyTorch's examples.imagenet.main
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_model(model, optim, iter, model_dir, max_to_keep=None, model_name=""):
    checkpoint = {
        'iter': iter,
        'model': model.state_dict(),
        'optim': optim.state_dict() if optim is not None else None}
    if model_name == "":
        torch.save(checkpoint, "{}/checkpoint_{:06}.pth".format(model_dir, iter))
    else:
        torch.save(checkpoint, "{}/{}_checkpoint_{:06}.pth".format(model_dir, model_name, iter))

    if max_to_keep is not None and max_to_keep > 0:
        checkpoint_list = sorted([ckpt for ckpt in glob.glob(model_dir + "/" + '*.pth')])
        while len(checkpoint_list) > max_to_keep:
            os.remove(checkpoint_list[0])
            checkpoint_list = checkpoint_list[1:]

def load_model(model, optim, iter, model_dir, model_name=''):
    if model_name == "":
        checkpoint = torch.load("{}/checkpoint_{:06}.pth".format(model_dir, iter))
    else:
        checkpoint = torch.load("{}/{}_checkpoint_{:06}.pth".format(model_dir, model_name, iter))

    model.load_state_dict(checkpoint['model'])
    if optim:
        optim.load_state_dict(checkpoint['optim'])


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def init_modules(modules, w_init='kaiming_uniform'):
    if w_init == "normal":
        _init = init.normal_
    elif w_init == "xavier_normal":
        _init = init.xavier_normal_
    elif w_init == "xavier_uniform":
        _init = init.xavier_uniform_
    elif w_init == "kaiming_normal":
        _init = init.kaiming_normal_
    elif w_init == "kaiming_uniform":
        _init = init.kaiming_uniform_
    elif w_init == "orthogonal":
        _init = init.orthogonal_
    else:
        raise NotImplementedError
    for m in modules:
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            _init(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        if isinstance(m, (nn.LSTM, nn.GRU)):
            for name, param in m.named_parameters():
                if 'bias' in name:
                    nn.init.zeros_(param)
                elif 'weight' in name:
                    _init(param)


def load_vocab(cfg):
    def invert_dict(d):
        return {v: k for k, v in d.items()}

    with open(os.path.join(cfg.DATASET.DATA_DIR, 'dic.pkl'), 'rb') as f:
        dictionaries = pickle.load(f)
    vocab = {}
    vocab['question_token_to_idx'] = dictionaries["word_dic"]
    vocab['answer_token_to_idx'] = dictionaries["answer_dic"]
    vocab['question_token_to_idx']['pad'] = 0
    vocab['question_idx_to_token'] = invert_dict(vocab['question_token_to_idx'])
    vocab['answer_idx_to_token'] = invert_dict(vocab['answer_token_to_idx'])

    return vocab


def get_labels_concepts_filename(cfg, get_concept_names_filename=False):
    label_names_filename = cfg.PREPROCESS.LABEL_NAMES_PATH.split('.')[0].split('/')[-1]
    suffix = '{}_{}_{}_{}'.format(
        label_names_filename,
        'tok' if cfg.PREPROCESS.WORD_TOKENIZE else 'ntok',
        'stop' if cfg.PREPROCESS.REMOVE_STOPWORDS else 'nstop',
        'lem' if cfg.PREPROCESS.LEMMATIZE else 'nlem',
    )
    labels_concepts_filename = os.path.join(cfg.DATASET.LABELS_CONCEPTS_DIR, 'lab_con_' + suffix)
    if get_concept_names_filename:
        concept_names_filename = os.path.join(cfg.DATASET.LABELS_CONCEPTS_DIR, 'concept_names_' + suffix)
        return labels_concepts_filename, concept_names_filename
    else:
        return labels_concepts_filename


def preprocess_sentence(cfg, sentence):
    # All letters to lowercase
    sentence = sentence.lower()
    # Tokenize
    tokens = word_tokenize(sentence) if cfg.PREPROCESS.WORD_TOKENIZE else sentence.split(' ')
    # Remove stopwords
    if cfg.PREPROCESS.REMOVE_STOPWORDS:
        tokens = list(filter(lambda x: x not in stopwords.words('english'), tokens))
    # Lemmatize (stemming)
    if cfg.PREPROCESS.LEMMATIZE:
        stemmer = PorterStemmer()
        tokens = list(map(stemmer.stem, tokens))

    return tokens

def load_label_word_ids(cfg):
    # Load labels list
    labels = [line.strip() for line in open(cfg.PREPROCESS.LABEL_NAMES_PATH)]

    label_words = []
    max_len = 0
    for label in labels:
        words = preprocess_sentence(cfg, label)
        if len(words) > max_len:
            max_len = len(words)
        for word in words:
            if word not in label_words:
                label_words.append(word)
    vocab_size = len(label_words) + 1
    
    # Create mapping from word to id
    #   Start from 1 to use 0 for padding
    word_to_id = { word: i+1 for i, word in enumerate(label_words) }

    labels_matrix = np.empty([len(labels), max_len], dtype=np.int64)
    for i, label in enumerate(labels):
        words = preprocess_sentence(cfg, label)
        ids = [word_to_id[word] for word in words]
        ids += [0] * (max_len - len(words))
        labels_matrix[i] = np.array(ids)

    concepts = []
    for word in label_words:
        concepts.append(word_to_id[word])
    concepts = np.array(concepts, dtype=np.int64)

    return labels_matrix, concepts, vocab_size


def load_label_embeddings(cfg, get_concept_words=False, get_concepts_per_label=False):
    def invert_dict(d):
        return {v: k for k, v in d.items()}

    # Load word embeddings
    word_to_vec = {}
    for line in open(cfg.PREPROCESS.GLOVE_PATH, encoding='utf8'):
        splitted = line.split(' ')
        word = splitted[0]
        vec = np.array([float(n) for n in splitted[1:]])
        word_to_vec[word] = vec

    # Load labels list
    labels = [line.strip() for line in open(cfg.PREPROCESS.LABEL_NAMES_PATH)]
    # Map labels to words
    labels = list(map(lambda label: preprocess_sentence(cfg, label), labels))
    embed_dim = next(iter(word_to_vec.values())).shape[0]

    # Create labels matrix
    concept_words = []
    labels_matrix = np.empty((len(labels), embed_dim), dtype=np.float32)
    for i, words in enumerate(labels):
        embed_sum = np.zeros(embed_dim)
        n_embed = 0
        for word in words:
            if word not in ['something']:
                n_embed += 1
                vec = word_to_vec.get(word)
                if vec is not None:
                    embed_sum += vec
                    if word not in concept_words:
                        concept_words.append(word)
        labels_matrix[i] = embed_sum / n_embed        

    # Create concepts
    concepts = []
    for word in concept_words:
        vec = word_to_vec.get(word)
        concepts.append(vec)
    concepts = np.array(concepts, dtype=np.float32)

    results = [labels_matrix, concepts]

    if get_concept_words:
        results.append(concept_words)

    if get_concepts_per_label:
        # Create a list of multi-hot concepts for each label
        concept_word_to_idx = {}
        for i, concept_word in enumerate(concept_words):
            concept_word_to_idx[concept_word] = i
        concepts_per_label = np.zeros((len(labels), len(concept_words)), dtype=np.float32)
        for i, words in enumerate(labels):
            concepts_idxs = list(filter(lambda x: x is not None, (concept_word_to_idx.get(word) for word in words)))
            concepts_per_label[i, concepts_idxs] = 1

        results.append(concepts_per_label)

    return results


def generateVarDpMask(shape, keepProb):
    randomTensor = torch.tensor(keepProb).cuda().expand(shape)
    randomTensor += nn.init.uniform_(torch.cuda.FloatTensor(shape[0], shape[1]))
    binaryTensor = torch.floor(randomTensor)
    mask = torch.cuda.FloatTensor(binaryTensor)
    return mask


def applyVarDpMask(inp, mask, keepProb):
    ret = (torch.div(inp, torch.tensor(keepProb).cuda())) * mask
    return ret

def calc_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def remove_module_from_checkpoint_state_dict(state_dict):
    """
    Removes the prefix `module` from weight names that gets added by
    torch.nn.DataParallel()
    """
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict