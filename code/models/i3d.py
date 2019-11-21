import math
import os

import numpy as np
import torch
from torch import nn
from torch.nn import ReplicationPad3d


def get_padding_shape(filter_shape, stride):
    def _pad_top_bottom(filter_dim, stride_val):
        pad_along = max(filter_dim - stride_val, 0)
        pad_top = pad_along // 2
        pad_bottom = pad_along - pad_top
        return pad_top, pad_bottom

    padding_shape = []
    for filter_dim, stride_val in zip(filter_shape, stride):
        pad_top, pad_bottom = _pad_top_bottom(filter_dim, stride_val)
        padding_shape.append(pad_top)
        padding_shape.append(pad_bottom)
    depth_top = padding_shape.pop(0)
    depth_bottom = padding_shape.pop(0)
    padding_shape.append(depth_top)
    padding_shape.append(depth_bottom)

    return tuple(padding_shape)


def simplify_padding(padding_shapes):
    all_same = True
    padding_init = padding_shapes[0]
    for pad in padding_shapes[1:]:
        if pad != padding_init:
            all_same = False
    return all_same, padding_init


def check_missing_keys(incompatible_keys):
    missing_keys = incompatible_keys.missing_keys
    if len(missing_keys) > 1:
        raise RuntimeError('Missing key(s) in state_dict: {}'.format(
            ', '.join('"{}"'.format(k) for k in missing_keys)
        ))


class Unit3Dpy(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=(1, 1, 1),
                 stride=(1, 1, 1),
                 activation='relu',
                 padding='SAME',
                 use_bias=False,
                 use_bn=True):
        super(Unit3Dpy, self).__init__()

        self.padding = padding
        self.activation = activation
        self.use_bn = use_bn
        if padding == 'SAME':
            padding_shape = get_padding_shape(kernel_size, stride)
            simplify_pad, pad_size = simplify_padding(padding_shape)
            self.simplify_pad = simplify_pad
        elif padding == 'VALID':
            padding_shape = 0
        else:
            raise ValueError(
                'padding should be in [VALID|SAME] but got {}'.format(padding))

        if padding == 'SAME':
            if not simplify_pad:
                self.pad = nn.ConstantPad3d(padding_shape, 0)
                self.conv3d = nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    bias=use_bias)
            else:
                self.conv3d = nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=pad_size,
                    bias=use_bias)
        elif padding == 'VALID':
            self.conv3d = nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding_shape,
                stride=stride,
                bias=use_bias)
        else:
            raise ValueError(
                'padding should be in [VALID|SAME] but got {}'.format(padding))

        if self.use_bn:
            self.batch3d = nn.BatchNorm3d(out_channels)

        if activation == 'relu':
            self.activation = nn.functional.relu

    def forward(self, inp):
        if self.padding == 'SAME' and self.simplify_pad is False:
            inp = self.pad(inp)
        out = self.conv3d(inp)
        if self.use_bn:
            out = self.batch3d(out)
        if self.activation is not None:
            out = nn.functional.relu(out)
        return out


class MaxPool3dTFPadding(nn.Module):
    def __init__(self, kernel_size, stride=None, padding='SAME'):
        super(MaxPool3dTFPadding, self).__init__()
        if padding == 'SAME':
            padding_shape = get_padding_shape(kernel_size, stride)
            self.padding_shape = padding_shape
            self.pad = nn.ConstantPad3d(padding_shape, 0)
        self.pool = nn.MaxPool3d(kernel_size, stride, ceil_mode=True)

    def forward(self, inp):
        inp = self.pad(inp)
        out = self.pool(inp)
        return out


class Mixed(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Mixed, self).__init__()
        # Branch 0
        self.branch_0 = Unit3Dpy(
            in_channels, out_channels[0], kernel_size=(1, 1, 1))

        # Branch 1
        branch_1_conv1 = Unit3Dpy(
            in_channels, out_channels[1], kernel_size=(1, 1, 1))
        branch_1_conv2 = Unit3Dpy(
            out_channels[1], out_channels[2], kernel_size=(3, 3, 3))
        self.branch_1 = nn.Sequential(branch_1_conv1, branch_1_conv2)

        # Branch 2
        branch_2_conv1 = Unit3Dpy(
            in_channels, out_channels[3], kernel_size=(1, 1, 1))
        branch_2_conv2 = Unit3Dpy(
            out_channels[3], out_channels[4], kernel_size=(3, 3, 3))
        self.branch_2 = nn.Sequential(branch_2_conv1, branch_2_conv2)

        # Branch3
        branch_3_pool = MaxPool3dTFPadding(
            kernel_size=(3, 3, 3), stride=(1, 1, 1), padding='SAME')
        branch_3_conv2 = Unit3Dpy(
            in_channels, out_channels[5], kernel_size=(1, 1, 1))
        self.branch_3 = nn.Sequential(branch_3_pool, branch_3_conv2)

    def forward(self, inp):
        out_0 = self.branch_0(inp)
        out_1 = self.branch_1(inp)
        out_2 = self.branch_2(inp)
        out_3 = self.branch_3(inp)
        out = torch.cat((out_0, out_1, out_2, out_3), 1)
        return out


class I3DBackbone(nn.Module):
    def __init__(self,
                 modality='rgb'):
        super().__init__()

        if modality == 'rgb':
            in_channels = 3
        elif modality == 'flow':
            in_channels = 2
        else:
            raise ValueError(
                '{} not among known modalities [rgb|flow]'.format(modality))
        self.modality = modality

        conv3d_1a_7x7 = Unit3Dpy(
            out_channels=64,
            in_channels=in_channels,
            kernel_size=(7, 7, 7),
            stride=(2, 2, 2),
            padding='SAME')
        # 1st conv-pool
        self.conv3d_1a_7x7 = conv3d_1a_7x7
        self.maxPool3d_2a_3x3 = MaxPool3dTFPadding(
            kernel_size=(1, 3, 3), stride=(1, 2, 2), padding='SAME')
        # conv conv
        conv3d_2b_1x1 = Unit3Dpy(
            out_channels=64,
            in_channels=64,
            kernel_size=(1, 1, 1),
            padding='SAME')
        self.conv3d_2b_1x1 = conv3d_2b_1x1
        conv3d_2c_3x3 = Unit3Dpy(
            out_channels=192,
            in_channels=64,
            kernel_size=(3, 3, 3),
            padding='SAME')
        self.conv3d_2c_3x3 = conv3d_2c_3x3
        self.maxPool3d_3a_3x3 = MaxPool3dTFPadding(
            kernel_size=(1, 3, 3), stride=(1, 2, 2), padding='SAME')

        # Mixed_3b
        self.mixed_3b = Mixed(192, [64, 96, 128, 16, 32, 32])
        self.mixed_3c = Mixed(256, [128, 128, 192, 32, 96, 64])

        self.maxPool3d_4a_3x3 = MaxPool3dTFPadding(
            kernel_size=(3, 3, 3), stride=(2, 2, 2), padding='SAME')

        # Mixed 4
        self.mixed_4b = Mixed(480, [192, 96, 208, 16, 48, 64])
        self.mixed_4c = Mixed(512, [160, 112, 224, 24, 64, 64])
        self.mixed_4d = Mixed(512, [128, 128, 256, 24, 64, 64])

    def forward(self, out):
        # Preprocessing
        out = self.conv3d_1a_7x7(out) # 64x36x112x112
        out = self.maxPool3d_2a_3x3(out) # 64x36x56x56
        out = self.conv3d_2b_1x1(out) # 64x36x56x56
        out = self.conv3d_2c_3x3(out) # 192x36x56x56
        out = self.maxPool3d_3a_3x3(out) # 192x36x28x28
        out = self.mixed_3b(out) # 256x36x28x28
        out = self.mixed_3c(out) # 480x36x28x28
        out = self.maxPool3d_4a_3x3(out) # 480x18x14x14
        out = self.mixed_4b(out) # 512x18x14x14
        out = self.mixed_4c(out) # 512x18x14x14
        out = self.mixed_4d(out) # 512x18x14x14

        return out


class I3DCommonFinetune(nn.Module):
    def __init__(self):
        super().__init__()

        self.mixed_4e = Mixed(512, [112, 144, 288, 32, 64, 64])
        self.mixed_4f = Mixed(528, [256, 160, 320, 32, 128, 128])

    def forward(self, out):
        out = self.mixed_4e(out) # 528x18x14x14
        out = self.mixed_4f(out) # 832x18x14x14

        return out


class I3DHeadFinetune(nn.Module):
    def __init__(self, dropout_prob=0):
        super().__init__()

        self.maxPool3d_5a_2x2 = MaxPool3dTFPadding(
            kernel_size=(2, 2, 2), stride=(2, 2, 2), padding='SAME')

        # Mixed 5
        self.mixed_5b = Mixed(832, [256, 160, 320, 32, 128, 128])
        self.mixed_5c = Mixed(832, [384, 192, 384, 48, 128, 128])

        self.avg_pool = nn.AvgPool3d((2, 7, 7), (1, 1, 1))
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, out):
        out = self.maxPool3d_5a_2x2(out) # 832x9x7x7
        out = self.mixed_5b(out)
        out = self.mixed_5c(out)
        out = self.avg_pool(out)
        out = self.dropout(out)

        return out
        

class I3DClassifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        if num_classes <= 0:
            raise ValueError('num_classes must be > 0')

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.conv3d_0c_1x1 = Unit3Dpy(
            in_channels=self.in_channels,
            out_channels=self.num_classes,
            kernel_size=(1, 1, 1),
            activation=None,
            use_bias=True,
            use_bn=False)
        self.softmax = nn.Softmax(1)

    def forward(self, out):
        out = self.conv3d_0c_1x1(out)
        out = out.squeeze(3)
        out = out.squeeze(3)
        out = out.mean(2)
        out_logits = out
        out = self.softmax(out_logits)

        return out

class I3D(nn.Module):
    def __init__(self,
                 num_classes,
                 modality='rgb',
                 dropout_prob=0,
                 name='inception'):
        super(I3D, self).__init__()

        self.name = name
        self.num_classes = num_classes
        if modality == 'rgb':
            in_channels = 3
        elif modality == 'flow':
            in_channels = 2
        else:
            raise ValueError(
                '{} not among known modalities [rgb|flow]'.format(modality))
        self.modality = modality

        conv3d_1a_7x7 = Unit3Dpy(
            out_channels=64,
            in_channels=in_channels,
            kernel_size=(7, 7, 7),
            stride=(2, 2, 2),
            padding='SAME')
        # 1st conv-pool
        self.conv3d_1a_7x7 = conv3d_1a_7x7
        self.maxPool3d_2a_3x3 = MaxPool3dTFPadding(
            kernel_size=(1, 3, 3), stride=(1, 2, 2), padding='SAME')
        # conv conv
        conv3d_2b_1x1 = Unit3Dpy(
            out_channels=64,
            in_channels=64,
            kernel_size=(1, 1, 1),
            padding='SAME')
        self.conv3d_2b_1x1 = conv3d_2b_1x1
        conv3d_2c_3x3 = Unit3Dpy(
            out_channels=192,
            in_channels=64,
            kernel_size=(3, 3, 3),
            padding='SAME')
        self.conv3d_2c_3x3 = conv3d_2c_3x3
        self.maxPool3d_3a_3x3 = MaxPool3dTFPadding(
            kernel_size=(1, 3, 3), stride=(1, 2, 2), padding='SAME')

        # Mixed_3b
        self.mixed_3b = Mixed(192, [64, 96, 128, 16, 32, 32])
        self.mixed_3c = Mixed(256, [128, 128, 192, 32, 96, 64])

        self.maxPool3d_4a_3x3 = MaxPool3dTFPadding(
            kernel_size=(3, 3, 3), stride=(2, 2, 2), padding='SAME')

        # Mixed 4
        self.mixed_4b = Mixed(480, [192, 96, 208, 16, 48, 64])
        self.mixed_4c = Mixed(512, [160, 112, 224, 24, 64, 64])
        self.mixed_4d = Mixed(512, [128, 128, 256, 24, 64, 64])
        self.mixed_4e = Mixed(512, [112, 144, 288, 32, 64, 64])
        self.mixed_4f = Mixed(528, [256, 160, 320, 32, 128, 128])

        self.maxPool3d_5a_2x2 = MaxPool3dTFPadding(
            kernel_size=(2, 2, 2), stride=(2, 2, 2), padding='SAME')

        # Mixed 5
        self.mixed_5b = Mixed(832, [256, 160, 320, 32, 128, 128])
        self.mixed_5c = Mixed(832, [384, 192, 384, 48, 128, 128])

        self.avg_pool = nn.AvgPool3d((2, 7, 7), (1, 1, 1))
        self.dropout = nn.Dropout(dropout_prob)
        self.conv3d_0c_1x1 = Unit3Dpy(
            in_channels=1024,
            out_channels=self.num_classes,
            kernel_size=(1, 1, 1),
            activation=None,
            use_bias=True,
            use_bn=False)
        self.softmax = nn.Softmax(1)

    def forward(self, inp):
        # Preprocessing
        out = self.conv3d_1a_7x7(inp) # 64x36x112x112
        out = self.maxPool3d_2a_3x3(out) # 64x36x56x56
        out = self.conv3d_2b_1x1(out) # 64x36x56x56
        out = self.conv3d_2c_3x3(out) # 192x36x56x56
        out = self.maxPool3d_3a_3x3(out) # 192x36x28x28
        out = self.mixed_3b(out) # 256x36x28x28
        out = self.mixed_3c(out) # 480x36x28x28
        out = self.maxPool3d_4a_3x3(out) # 480x18x14x14
        out = self.mixed_4b(out) # 512x18x14x14
        out = self.mixed_4c(out) # 512x18x14x14
        out = self.mixed_4d(out) # 512x18x14x14
        out = self.mixed_4e(out) # 528x18x14x14
        out = self.mixed_4f(out) # 832x18x14x14
        out = self.maxPool3d_5a_2x2(out) # 832x9x7x7
        out = self.mixed_5b(out)
        out = self.mixed_5c(out)
        out = self.avg_pool(out)
        out = self.dropout(out)
        out = self.conv3d_0c_1x1(out)
        out = out.squeeze(3)
        out = out.squeeze(3)
        out = out.mean(2)
        out_logits = out
        out = self.softmax(out_logits)
        return out, out_logits

    def load_tf_weights(self, sess):
        state_dict = {}
        if self.modality == 'rgb':
            prefix = 'RGB/inception_i3d'
        elif self.modality == 'flow':
            prefix = 'Flow/inception_i3d'
        load_conv3d(state_dict, 'conv3d_1a_7x7', sess,
                    os.path.join(prefix, 'Conv3d_1a_7x7'))
        load_conv3d(state_dict, 'conv3d_2b_1x1', sess,
                    os.path.join(prefix, 'Conv3d_2b_1x1'))
        load_conv3d(state_dict, 'conv3d_2c_3x3', sess,
                    os.path.join(prefix, 'Conv3d_2c_3x3'))

        load_mixed(state_dict, 'mixed_3b', sess,
                   os.path.join(prefix, 'Mixed_3b'))
        load_mixed(state_dict, 'mixed_3c', sess,
                   os.path.join(prefix, 'Mixed_3c'))
        load_mixed(state_dict, 'mixed_4b', sess,
                   os.path.join(prefix, 'Mixed_4b'))
        load_mixed(state_dict, 'mixed_4c', sess,
                   os.path.join(prefix, 'Mixed_4c'))
        load_mixed(state_dict, 'mixed_4d', sess,
                   os.path.join(prefix, 'Mixed_4d'))
        load_mixed(state_dict, 'mixed_4e', sess,
                   os.path.join(prefix, 'Mixed_4e'))
        # Here goest to 0.1 max error with tf
        load_mixed(state_dict, 'mixed_4f', sess,
                   os.path.join(prefix, 'Mixed_4f'))

        load_mixed(
            state_dict,
            'mixed_5b',
            sess,
            os.path.join(prefix, 'Mixed_5b'),
            fix_typo=True)
        load_mixed(state_dict, 'mixed_5c', sess,
                   os.path.join(prefix, 'Mixed_5c'))
        load_conv3d(
            state_dict,
            'conv3d_0c_1x1',
            sess,
            os.path.join(prefix, 'Logits', 'Conv3d_0c_1x1'),
            bias=True,
            bn=False)
        self.load_state_dict(state_dict)


class I3DFinetune(nn.Module):
    def __init__(self,
                 num_classes,
                 modality='rgb',
                 dropout_prob=0,
                 name='inception'):
        super().__init__()

        self.name = name
        self.backbone_module = I3DBackbone(modality=modality)
        for param in self.backbone_module.parameters():
            param.requires_grad = False
        self.finetune_module1 = I3DCommonFinetune()
        self.finetune_module2 = I3DHeadFinetune(dropout_prob=dropout_prob)
        self.clf = I3DClassifier(1024, num_classes)

    def forward(self, out):
        out = self.backbone_module(out)
        out = self.finetune_module1(out)
        out = self.finetune_module2(out)
        out = self.clf(out)

        return out

    def load_state_dict_from_i3d(self, state_dict):
        # Load backbone state dict
        incompatible_keys = self.backbone_module.load_state_dict(state_dict, strict=False)
        check_missing_keys(incompatible_keys)
        # Load common finetune state dict
        incompatible_keys = self.finetune_module1.load_state_dict(state_dict, strict=False)
        check_missing_keys(incompatible_keys)
        # Load head finetune state dicts
        incompatible_keys = self.finetune_module2.load_state_dict(state_dict, strict=False)
        check_missing_keys(incompatible_keys)    


class I3DMultiHead(nn.Module):
    def __init__(self,
                num_classes_list,
                final_num_classes,
                modality='rgb',
                dropout_prob=0,
                name='inception'):
        super().__init__()

        self.name = name
        self.n_heads = len(num_classes_list)
        self.backbone_module = I3DBackbone(modality=modality)
        for param in self.backbone_module.parameters():
            param.requires_grad = False
        self.common_finetune_module = I3DCommonFinetune()
        self.head_modules = nn.ModuleList()
        self.head_clfs = nn.ModuleList()
        for num_classes in num_classes_list:
            self.head_modules.append(I3DHeadFinetune(dropout_prob=dropout_prob))
            clf = I3DClassifier(1024, num_classes)
            self.head_clfs.append(clf)

        self.final_clf = I3DClassifier(1024 * len(num_classes_list), final_num_classes)

    def forward(self, out):
        out = self.backbone_module(out)
        out = self.common_finetune_module(out)
        head_feats_list = []
        head_out_list = []
        for i in range(self.n_heads):
            head_feats = self.head_modules[i](out)
            head_feats_list.append(head_feats)
            head_out_list.append(self.head_clfs[i](head_feats))

        final_out = self.final_clf(torch.cat(head_feats_list, dim=1))

        return head_out_list, final_out

    def load_state_dict_from_i3d(self, state_dict):
        # Load backbone state dict
        incompatible_keys = self.backbone_module.load_state_dict(state_dict, strict=False)
        check_missing_keys(incompatible_keys)
        # Load common finetune state dict
        incompatible_keys = self.common_finetune_module.load_state_dict(state_dict, strict=False)
        check_missing_keys(incompatible_keys)
        # Load head finetune state dicts
        for head in self.head_modules:
            incompatible_keys = head.load_state_dict(state_dict, strict=False)
            check_missing_keys(incompatible_keys)


def get_conv_params(sess, name, bias=False):
    # Get conv weights
    conv_weights_tensor = sess.graph.get_tensor_by_name(
        os.path.join(name, 'w:0'))
    if bias:
        conv_bias_tensor = sess.graph.get_tensor_by_name(
            os.path.join(name, 'b:0'))
        conv_bias = sess.run(conv_bias_tensor)
    conv_weights = sess.run(conv_weights_tensor)
    conv_shape = conv_weights.shape

    kernel_shape = conv_shape[0:3]
    in_channels = conv_shape[3]
    out_channels = conv_shape[4]

    conv_op = sess.graph.get_operation_by_name(
        os.path.join(name, 'convolution'))
    padding_name = conv_op.get_attr('padding')
    padding = _get_padding(padding_name, kernel_shape)
    all_strides = conv_op.get_attr('strides')
    strides = all_strides[1:4]
    conv_params = [
        conv_weights, kernel_shape, in_channels, out_channels, strides, padding
    ]
    if bias:
        conv_params.append(conv_bias)
    return conv_params


def get_bn_params(sess, name):
    moving_mean_tensor = sess.graph.get_tensor_by_name(
        os.path.join(name, 'moving_mean:0'))
    moving_var_tensor = sess.graph.get_tensor_by_name(
        os.path.join(name, 'moving_variance:0'))
    beta_tensor = sess.graph.get_tensor_by_name(os.path.join(name, 'beta:0'))
    moving_mean = sess.run(moving_mean_tensor)
    moving_var = sess.run(moving_var_tensor)
    beta = sess.run(beta_tensor)
    return moving_mean, moving_var, beta


def _get_padding(padding_name, conv_shape):
    padding_name = padding_name.decode("utf-8")
    if padding_name == "VALID":
        return [0, 0]
    elif padding_name == "SAME":
        # return [math.ceil(int(conv_shape[0])/2), math.ceil(int(conv_shape[1])/2)]
        return [
            math.floor(int(conv_shape[0]) / 2),
            math.floor(int(conv_shape[1]) / 2),
            math.floor(int(conv_shape[2]) / 2)
        ]
    else:
        raise ValueError('Invalid padding name ' + padding_name)


def load_conv3d(state_dict, name_pt, sess, name_tf, bias=False, bn=True):
    # Transfer convolution params
    conv_name_tf = os.path.join(name_tf, 'conv_3d')
    conv_params = get_conv_params(sess, conv_name_tf, bias=bias)
    if bias:
        conv_weights, kernel_shape, in_channels, out_channels, strides, padding, conv_bias = conv_params
    else:
        conv_weights, kernel_shape, in_channels, out_channels, strides, padding = conv_params

    conv_weights_rs = np.transpose(
        conv_weights, (4, 3, 0, 1,
                       2))  # to pt format (out_c, in_c, depth, height, width)
    state_dict[name_pt + '.conv3d.weight'] = torch.from_numpy(conv_weights_rs)
    if bias:
        state_dict[name_pt + '.conv3d.bias'] = torch.from_numpy(conv_bias)

    # Transfer batch norm params
    if bn:
        conv_tf_name = os.path.join(name_tf, 'batch_norm')
        moving_mean, moving_var, beta = get_bn_params(sess, conv_tf_name)

        out_planes = conv_weights_rs.shape[0]
        state_dict[name_pt + '.batch3d.weight'] = torch.ones(out_planes)
        state_dict[name_pt +
                   '.batch3d.bias'] = torch.from_numpy(beta.squeeze())
        state_dict[name_pt
                   + '.batch3d.running_mean'] = torch.from_numpy(moving_mean.squeeze())
        state_dict[name_pt
                   + '.batch3d.running_var'] = torch.from_numpy(moving_var.squeeze())


def load_mixed(state_dict, name_pt, sess, name_tf, fix_typo=False):
    # Branch 0
    load_conv3d(state_dict, name_pt + '.branch_0', sess,
                os.path.join(name_tf, 'Branch_0/Conv3d_0a_1x1'))

    # Branch .1
    load_conv3d(state_dict, name_pt + '.branch_1.0', sess,
                os.path.join(name_tf, 'Branch_1/Conv3d_0a_1x1'))
    load_conv3d(state_dict, name_pt + '.branch_1.1', sess,
                os.path.join(name_tf, 'Branch_1/Conv3d_0b_3x3'))

    # Branch 2
    load_conv3d(state_dict, name_pt + '.branch_2.0', sess,
                os.path.join(name_tf, 'Branch_2/Conv3d_0a_1x1'))
    if fix_typo:
        load_conv3d(state_dict, name_pt + '.branch_2.1', sess,
                    os.path.join(name_tf, 'Branch_2/Conv3d_0a_3x3'))
    else:
        load_conv3d(state_dict, name_pt + '.branch_2.1', sess,
                    os.path.join(name_tf, 'Branch_2/Conv3d_0b_3x3'))

    # Branch 3
    load_conv3d(state_dict, name_pt + '.branch_3.1', sess,
                os.path.join(name_tf, 'Branch_3/Conv3d_0b_1x1'))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['default', 'multihead'], default='default')
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if args.mode == 'default':
        model = I3D(num_classes=10, modality='rgb').to(device)
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        n_params = sum([np.prod(p.size()) for p in model_parameters])
        print('I3D number of params:', n_params)

        inp = torch.empty([2, 3, 10, 224, 224])
        out, out_logits = model(inp)

        print(out.shape)
    elif args.mode == 'multihead':
        model = I3DMultiHead(num_classes_list=[10, 5, 7])
        state_dict = torch.load('data/i3d_model_rgb.pth')
        model.load_state_dict_from_i3d(state_dict)

        image = torch.empty([2, 3, 10, 224, 224]).to(device)
        target = torch.empty([2]).to(device)

        out_list = model(image)
        print(list(map(lambda x: x.shape, out_list)))