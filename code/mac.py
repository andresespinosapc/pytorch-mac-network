import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable

from utils import *
from models.rgb_resnet import rgb_resnet50


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_MAC(cfg):
    kwargs = {
        'max_step': cfg.TRAIN.MAX_STEPS
    }

    model = MACNetwork(cfg, **kwargs)
    model_ema = MACNetwork(cfg, **kwargs)
    for param in model_ema.parameters():
        param.requires_grad = False

    if torch.cuda.is_available():
        model.cuda()
        model_ema.cuda()
    else:
        model.cpu()
        model_ema.cpu()
    model.train()
    return model, model_ema


class ControlUnit(nn.Module):
    def __init__(self, cfg, module_dim, max_step=4):
        super().__init__()
        self.cfg = cfg
        self.attn = nn.Linear(module_dim, 1)
        self.control_input = nn.Sequential(nn.Linear(module_dim, module_dim),
                                           nn.Tanh())

        self.control_input_u = nn.ModuleList()
        for i in range(max_step):
            self.control_input_u.append(nn.Linear(module_dim, module_dim))

        self.module_dim = module_dim

    def forward(self, memory, step):
        """
        Args:
            memory: last memory vector.
            control: previous control state
            step: which step in the reasoning chain
        """
        # compute interactions with question words
        # question = self.control_input(question)
        next_control = self.control_input_u[step](memory)

        # newContControl = memory
        # newContControl = torch.unsqueeze(newContControl, 1)
        # interactions = newContControl * context

        # # compute attention distribution over words and summarize them accordingly
        # logits = self.attn(interactions)

        # attn = F.softmax(logits, 1)

        # # apply soft attention to current context words
        # next_control = (attn * context).sum(1)

        return next_control


class ReadUnit(nn.Module):
    def __init__(self, module_dim):
        super().__init__()

        self.concat = nn.Linear(module_dim * 2, module_dim)
        self.concat_2 = nn.Linear(module_dim, module_dim)
        self.attn = nn.Linear(module_dim, 1)
        self.dropout = nn.Dropout(0.15)
        self.kproj = nn.Linear(module_dim, module_dim)
        self.mproj = nn.Linear(module_dim, module_dim)

        self.activation = nn.ELU()
        self.module_dim = module_dim

    def forward(self, memory, know, control, memDpMask=None):
        """
        Args:
            memory: the cell's memory state
                [batchSize, memDim]

            know: representation of the knowledge base (image).
                [batchSize, kbSize (Height * Width), memDim]

            control: the cell's control state
                [batchSize, ctrlDim]

            memDpMask: variational dropout mask (if used)
                [batchSize, memDim]
        """
        ## Step 1: knowledge base / memory interactions
        # compute interactions between knowledge base and memory
        know = self.dropout(know)
        if memDpMask is not None:
            if self.training:
                memory = applyVarDpMask(memory, memDpMask, 0.85)
        else:
            memory = self.dropout(memory)
        know_proj = self.kproj(know)
        memory_proj = self.mproj(memory)
        memory_proj = memory_proj.unsqueeze(1)
        interactions = know_proj * memory_proj

        # project memory interactions back to hidden dimension
        interactions = torch.cat([interactions, know_proj], -1)
        interactions = self.concat(interactions)
        interactions = self.activation(interactions)
        interactions = self.concat_2(interactions)

        ## Step 2: compute interactions with control
        control = control.unsqueeze(1)
        interactions = interactions * control
        interactions = self.activation(interactions)

        ## Step 3: sum attentions up over the knowledge base
        # transform vectors to attention distribution
        interactions = self.dropout(interactions)
        attn = self.attn(interactions).squeeze(-1)
        attn = F.softmax(attn, 1)

        # sum up the knowledge base according to the distribution
        attn = attn.unsqueeze(-1)
        read = (attn * know).sum(1)

        return read


class WriteUnit(nn.Module):
    def __init__(self, cfg, module_dim):
        super().__init__()
        self.cfg = cfg
        self.linear = nn.Linear(module_dim * 2, module_dim)

    def forward(self, memory, info):
        newMemory = torch.cat([memory, info], -1)
        newMemory = self.linear(newMemory)

        return newMemory


class MACUnit(nn.Module):
    def __init__(self, cfg, module_dim=512, max_step=4):
        super().__init__()
        self.cfg = cfg
        self.control = ControlUnit(cfg, module_dim, max_step+1)
        self.read = ReadUnit(module_dim)
        self.write = WriteUnit(cfg, module_dim)

        self.initial_memory = nn.Parameter(torch.zeros(1, module_dim))

        self.module_dim = module_dim
        self.max_step = max_step

    def zero_state(self, batch_size):
        initial_memory = self.initial_memory.expand(batch_size, self.module_dim)

        if self.cfg.TRAIN.VAR_DROPOUT:
            memDpMask = generateVarDpMask((batch_size, self.module_dim), 0.85)
        else:
            memDpMask = None

        return initial_memory, memDpMask

    def forward(self, knowledge):
        batch_size = knowledge.size(0)
        memory, memDpMask = self.zero_state(batch_size)
        control = self.control(memory, 0)

        for i in range(self.max_step):
            # control unit
            control = self.control(memory, i+1)
            # read unit
            info = self.read(memory, knowledge, control, memDpMask)
            # write unit
            memory = self.write(memory, info)

        return memory


class InputUnit(nn.Module):
    def __init__(self, cfg, wordvec_dim=300, rnn_dim=512, module_dim=512, bidirectional=True):
        super(InputUnit, self).__init__()

        self.dim = module_dim
        self.cfg = cfg

        self.stem = nn.Sequential(nn.Dropout(p=0.18),
                                  nn.Conv2d(1024, module_dim, 3, 1, 1),
                                  nn.ELU(),
                                  nn.Dropout(p=0.18),
                                  nn.Conv2d(module_dim, module_dim, kernel_size=3, stride=1, padding=1),
                                  nn.ELU())

        self.bidirectional = bidirectional
        if bidirectional:
            rnn_dim = rnn_dim // 2

        self.encoder = nn.LSTM(wordvec_dim, rnn_dim, batch_first=True, bidirectional=bidirectional)
        self.embedding_dropout = nn.Dropout(p=0.15)

    def forward(self, image):
        b_size = image.size(0)

        # get image features
        img = self.stem(image)
        img = img.view(b_size, self.dim, -1)
        img = img.permute(0,2,1)

        return img


class OutputUnit(nn.Module):
    def __init__(self, module_dim=512, num_answers=28):
        super(OutputUnit, self).__init__()

        self.classifier = nn.Sequential(nn.Dropout(0.15),
                                        nn.Linear(module_dim, module_dim),
                                        nn.ELU(),
                                        nn.Dropout(0.15),
                                        nn.Linear(module_dim, num_answers))

    def forward(self, memory):
        out = memory
        out = self.classifier(out)

        return out


class MACNetwork(nn.Module):
    def __init__(self, cfg, max_step):
        super().__init__()

        self.cfg = cfg

        self.input_unit = InputUnit(cfg)

        self.output_unit = OutputUnit(num_answers=cfg.TRAIN.NUM_CLASSES)

        self.mac = MACUnit(cfg, max_step=max_step)

        init_modules(self.modules(), w_init=self.cfg.TRAIN.WEIGHT_INIT)
        nn.init.normal_(self.mac.initial_memory)

    def forward(self, image):
        # get image embedding
        img = self.input_unit(image)

        # apply MacCell
        memory = self.mac(img)

        # get classification
        out = self.output_unit(memory)

        return out
