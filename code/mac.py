import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable

from utils import *


def load_MAC(cfg, vocab, labels_matrix, concepts):
    kwargs = {'vocab': vocab,
              'max_step': cfg.TRAIN.MAX_STEPS,
              'labels_matrix': labels_matrix,
              'concepts': concepts,
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
        self.control_input = nn.Sequential(nn.Linear(2 * module_dim, module_dim),
                                           nn.Tanh())

        self.control_input_u = nn.ModuleList()
        for i in range(max_step):
            self.control_input_u.append(nn.Linear(module_dim, module_dim))

        self.module_dim = module_dim

    def mask(self, question_lengths, device):
        max_len = question_lengths.max().item()
        mask = torch.arange(max_len, device=device).expand(len(question_lengths), int(max_len)) < question_lengths.unsqueeze(1)
        mask = mask.float()
        ones = torch.ones_like(mask)
        mask = (ones - mask) * (1e-30)
        return mask

    def forward(self, prev_control, prev_memory, context, step):
        """
        Args:
            question: external inputs to control unit (the question vector).
                [batchSize, ctrlDim]
            context: the representation of the words used to compute the attention.
                [batchSize, questionLength, ctrlDim]
            control: previous control state
            question_lengths: the length of each question.
                [batchSize]
            step: which step in the reasoning chain
        """
        # compute interactions with question words
        question = torch.cat([prev_control, prev_memory], dim=1)
        question = self.control_input(question)
        question = self.control_input_u[step](question)

        newContControl = question
        newContControl = torch.unsqueeze(newContControl, 1)
        interactions = newContControl * context

        # compute attention distribution over words and summarize them accordingly
        logits = self.attn(interactions)

        # TODO: add mask again?!
        # question_lengths = torch.cuda.FloatTensor(question_lengths)
        # mask = self.mask(question_lengths, logits.device).unsqueeze(-1)
        # logits += mask
        attn = F.softmax(logits, 1)

        # apply soft attention to current context words
        next_control = (attn * context).sum(1)

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
    def __init__(self, cfg, max_step=4):
        super().__init__()
        self.cfg = cfg
        module_dim = cfg.MODEL.MODULE_DIM
        self.control = ControlUnit(cfg, module_dim, max_step)
        self.read = ReadUnit(module_dim)
        self.write = WriteUnit(cfg, module_dim)

        self.initial_memory = nn.Parameter(torch.zeros(1, module_dim))
        self.initial_control = nn.Parameter(torch.zeros(1, module_dim))

        self.module_dim = module_dim
        self.max_step = max_step

    def zero_state(self, batch_size):
        initial_memory = self.initial_memory.expand(batch_size, self.module_dim)
        initial_control = self.initial_control.expand(batch_size, self.module_dim)

        if self.cfg.TRAIN.VAR_DROPOUT:
            memDpMask = generateVarDpMask((batch_size, self.module_dim), 0.85)
        else:
            memDpMask = None

        return initial_control, initial_memory, memDpMask

    def forward(self, context, knowledge):
        batch_size = context.size(0)
        control, memory, memDpMask = self.zero_state(batch_size)

        for i in range(self.max_step):
            # control unit
            control = self.control(control, memory, context, i)
            # read unit
            info = self.read(memory, knowledge, control, memDpMask)
            # write unit
            memory = self.write(memory, info)

        return memory


class InputUnit(nn.Module):
    def __init__(self, cfg, vocab_size, wordvec_dim=300, rnn_dim=512, bidirectional=True):
        super(InputUnit, self).__init__()

        module_dim = cfg.MODEL.MODULE_DIM
        self.dim = module_dim
        self.cfg = cfg

        self.stem = nn.Sequential(
            nn.Dropout(p=cfg.DROPOUT.STEM),
            nn.Conv3d(256, module_dim, 3, 1, 1),
            nn.ELU(),
            nn.Dropout(p=cfg.DROPOUT.STEM),
            nn.Conv3d(module_dim, module_dim, 3, 1, 1),
            nn.ELU(),
        )

        # self.encoder_embed = nn.Embedding(vocab_size, wordvec_dim)
        # self.embedding_dropout = nn.Dropout(p=0.15)

    def forward(self, image, question):
        b_size = question.size(0)

        # get image features
        img = self.stem(image)
        img = img.view(b_size, self.dim, -1)
        img = img.permute(0,2,1)

        # get question and contextual word embeddings
        # embed = self.encoder_embed(question)
        # embed = self.embedding_dropout(embed)
        contextual_words = question

        return contextual_words, img


class OutputUnit(nn.Module):
    def __init__(self, labels_matrix, wordvec_dim=300, num_answers=28):
        super(OutputUnit, self).__init__()

        module_dim = cfg.MODEL.MODULE_DIM
        self.memory_proj = nn.Linear(module_dim, wordvec_dim)

        # self.classifier = nn.Sequential(nn.Dropout(0.15),
        #                                 nn.Linear(module_dim * 2, module_dim),
        #                                 nn.ELU(),
        #                                 nn.Dropout(0.15),
        #                                 nn.Linear(module_dim, num_answers))

        self.labels_matrix = labels_matrix
        self.attn = nn.Linear(module_dim, 1)

    def forward(self, memory):
        # apply classifier to output of MacCell and the question
        memory = self.memory_proj(memory).unsqueeze(1)
        # out = self.classifier(out)
        
        labels_matrix = self.labels_matrix.unsqueeze(0).expand(memory.size(0), -1, -1)
        interactions = memory * labels_matrix
        logits = self.attn(interactions)
        out = F.softmax(logits, 1).squeeze(2)

        return out


class MACNetwork(nn.Module):
    def __init__(self, cfg, max_step, vocab, labels_matrix, concepts):
        super().__init__()

        self.cfg = cfg
        encoder_vocab_size = len(vocab['question_token_to_idx'])

        self.input_unit = InputUnit(cfg, vocab_size=encoder_vocab_size)

        self.output_unit = OutputUnit(torch.tensor(labels_matrix))

        self.mac = MACUnit(cfg, max_step=max_step)

        init_modules(self.modules(), w_init=self.cfg.TRAIN.WEIGHT_INIT)
        # nn.init.uniform_(self.input_unit.encoder_embed.weight, -1.0, 1.0)
        nn.init.normal_(self.mac.initial_memory)
        nn.init.normal_(self.mac.initial_control)

        self.concepts = torch.tensor(concepts).unsqueeze(0)

    def forward(self, image):
        # get image, word, and sentence embeddings
        contextual_words, img = self.input_unit(image, self.concepts.expand(image.size(0), -1, -1))

        # apply MacCell
        memory = self.mac(contextual_words, img)

        # get classification
        out = self.output_unit(memory)

        return out

if __name__ == '__main__':
    from types import SimpleNamespace

    def dump_to_namespace(ns, d):
        for k, v in d.items():
            if isinstance(v, dict):
                leaf_ns = SimpleNamespace()
                ns.__dict__[k] = leaf_ns
                dump_to_namespace(leaf_ns, v)
            else:
                ns.__dict__[k] = v

    cfg_dict = {
        'TRAIN': {
            'VAR_DROPOUT': False,
            'WEIGHT_INIT': 'xavier_uniform',
            'MAX_STEPS': 4,
        },
        'MODEL': {
            'MODULE_DIM': 300,
        },
        'DROPOUT': {
            'STEM': 0.18,
        }
    }

    cfg = SimpleNamespace()
    dump_to_namespace(cfg, cfg_dict)

    vocab = { 'question_token_to_idx': [] }
    labels_matrix = torch.empty([10, 300])
    concepts = torch.empty([20, 300])
    model = MACNetwork(cfg, 5, vocab, labels_matrix, concepts)

    image = torch.empty([2, 256, 72, 11, 11])
    target = torch.empty([2])
    scores = model(image)
    print(scores.shape)
