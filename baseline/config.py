from __future__ import division
from __future__ import print_function

import os.path as osp
import numpy as np
from easydict import EasyDict as edict


__C = edict()
cfg = __C

__C.GPU_ID = '0'
__C.CUDA = True
__C.WORKERS = 4

# Preprocess options
__C.PREPROCESS = edict()
__C.PREPROCESS.GLOVE_PATH = 'data/glove.840B.300d.txt'
__C.PREPROCESS.LABEL_NAMES_PATH = 'data/label_names.txt'
__C.PREPROCESS.WORD_TOKENIZE = False
__C.PREPROCESS.REMOVE_STOPWORDS = False
__C.PREPROCESS.LEMMATIZE = False
__C.PREPROCESS = dict(__C.PREPROCESS)

# Training options
__C.TRAIN = edict()
__C.TRAIN.FLAG = True
__C.TRAIN.OPTIMIZER = 'adam'
__C.TRAIN.MOMENTUM = 0.9
__C.TRAIN.WEIGHT_DECAY = 0.00001
__C.TRAIN.LEARNING_RATE = 0.0001
__C.TRAIN.EFFECTIVE_BATCH_SIZE = 66
__C.TRAIN.REAL_BATCH_SIZE = 22
__C.TRAIN.VAL_BATCH_SIZE = 200
__C.TRAIN.MAX_EPOCHS = 25
__C.TRAIN.SNAPSHOT_INTERVAL = 5
__C.TRAIN.RESUME_SNAPSHOT_DIR = ""
__C.TRAIN.RESUME_SNAPSHOT_ITER = 0
__C.TRAIN.WEIGHT_INIT = "xavier_uniform"
__C.TRAIN.CLIP_GRADS = True
__C.TRAIN.CLIP = 8
__C.TRAIN.MAX_STEPS = 4
__C.TRAIN.EARLY_STOPPING = True
__C.TRAIN.PATIENCE = 5
__C.TRAIN.VAR_DROPOUT = False
__C.TRAIN = dict(__C.TRAIN)

__C.MODEL = edict()
__C.MODEL.CONCEPT_AUX_TASK = False
__C.MODEL.CONCEPT_AUX_WEIGHT = 0.5
__C.MODEL.USE_BASELINE_BACKBONE = False
__C.MODEL.BASELINE_BACKBONE_CHECKPOINT = ''
__C.MODEL.STEM = 'from_mac'
__C.MODEL.STEM_DROPOUT3D = True
__C.MODEL.STEM_BATCHNORM = True
__C.MODEL.MODULE_DIM = 512
__C.MODEL.GLOVE_LINEAR = True
__C.MODEL = dict(__C.MODEL)

__C.DROPOUT = edict()
__C.DROPOUT.READ_UNIT = 0.15
__C.DROPOUT.STEM = 0.18
__C.DROPOUT = dict(__C.DROPOUT)

# Dataset options
__C.DATASET = edict()
__C.DATASET.DATA_TYPE = 'features'
__C.DATASET.DATA_FOLDER = ''
__C.DATASET.TRAIN_JSON_PATH = ''
__C.DATASET.VAL_JSON_PATH = ''
__C.DATASET.LABELS_JSON_PATH = ''
__C.DATASET.CLIP_SIZE = 72
__C.DATASET.NCLIPS_TRAIN = 1
__C.DATASET.NCLIPS_VAL = 1
__C.DATASET.STEP_SIZE_TRAIN = 1
__C.DATASET.STEP_SIZE_VAL = 1
__C.DATASET.INPUT_SPATIAL_SIZE = 224
__C.DATASET.UPSCALE_FACTOR_TRAIN = 1.1
__C.DATASET.UPSCALE_FACTOR_EVAL = 1.0
__C.DATASET.FEATURES_PATH = 'data/feats.h5'
__C.DATASET.LABELS_CONCEPTS_DIR = 'data'
__C.DATASET.LEARNED_EMBEDDINGS = False
__C.DATASET = dict(__C.DATASET)


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if not k in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            elif isinstance(b[k], list):
                v = v.split(",")
                v = [int(_v) for _v in v]
            elif b[k] is None:
                if v == "None":
                    continue
                else:
                    v = v
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)
