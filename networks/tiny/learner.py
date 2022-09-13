### Reproduce of La-MAML from https://github.com/montrealrobotics/La-MAML

import math
import os
import sys
import traceback
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

class Learner(nn.Module):

    def __init__(self, config, args=None):
        """

        :param config: network config file, type:list of (string, list)
        :param imgc: 1 or 3
        :param imgsz:  28 or 84
        """
        super(Learner, self).__init__()

        self.config = config
        self.tf_counter = 0
        self.args = args

        # this dict contains all tensors needed to be optimized
        self.vars = nn.ParameterList()
        # running_mean and running_var
        self.vars_bn = nn.ParameterList()

        self.names = []

        for i, (name, param, extra_name) in enumerate(self.config):
            if name == 'conv2d':
                # [ch_out, ch_in, kernelsz, kernelsz
                w = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

            elif name == 'conv2d-nbias':
                # [ch_out, ch_in, kernelsz, kernelsz]
                w = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)

            elif name == 'linear':
                # layer += 1
                # [ch_out, ch_in]
                w = nn.Parameter(torch.ones(*param))
                # gain=1 according to cbfinn's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

            elif name == 'linear-nbias':
                # layer += 1
                # [ch_out, ch_in]
                w = nn.Parameter(torch.ones(*param))
                # gain=1 according to cbfinn's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]

            elif name == 'head-nbias':
                # layer += 1
                # [ch_out, ch_in]
                w = nn.Parameter(torch.ones(*param))
                # gain=1 according to cbfinn's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]

            elif name == 'cat':
                pass
            elif name == 'cat_start':
                pass
            elif name == "rep":
                pass
            elif name in ["residual3", "residual5", "in"]:
                pass
            elif name == 'bn':
                # [ch_out]
                w = nn.Parameter(torch.ones(param[0]))
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

                # must set requires_grad=False
                if self.args.use_track:
                    running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
                    running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
                    self.vars_bn.extend([running_mean, running_var])
                else:
                    self.vars_bn = None

            elif name in ['tanh', 'relu', 'upsample', 'avg_pool2d', 'max_pool2d',
                          'flatten', 'reshape', 'leakyrelu', 'sigmoid']:
                continue
            else:
                raise NotImplementedError

    def extra_repr(self):

        info = ''

        for name, param, extra_name in self.config:
            if name == 'conv2d':
                tmp = 'conv2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)' \
                      % (param[1], param[0], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name == 'conv2d-nbias':
                tmp = 'conv2d-nbias:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)' \
                      % (param[1], param[0], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name == 'linear':
                tmp = 'linear:(in:%d, out:%d)' % (param[1], param[0])
                info += tmp + '\n'

            elif name == 'linear-nbias':
                tmp = 'linear-nbias:(in:%d, out:%d)' % (param[1], param[0])
                info += tmp + '\n'

            elif name == 'head-nbias':
                tmp = 'head-nbias:(in:%d, out:%d)' % (param[1], param[0])
                info += tmp + '\n'

            elif name == 'leakyrelu':
                tmp = 'leakyrelu:(slope:%f)' % (param[0])
                info += tmp + '\n'

            elif name == 'cat':
                tmp = 'cat'
                info += tmp + "\n"
            elif name == 'cat_start':
                tmp = 'cat_start'
                info += tmp + "\n"

            elif name == 'rep':
                tmp = 'rep'
                info += tmp + "\n"

            elif name == 'avg_pool2d':
                tmp = 'avg_pool2d:(k:%d, stride:%d, padding:%d)' % (param[0], param[1], param[2])
                info += tmp + '\n'
            elif name == 'max_pool2d':
                tmp = 'max_pool2d:(k:%d, stride:%d, padding:%d)' % (param[0], param[1], param[2])
                info += tmp + '\n'
            elif name in ['flatten', 'tanh', 'relu', 'upsample', 'reshape', 'sigmoid', 'use_logits', 'bn']:
                tmp = name + ':' + str(tuple(param))
                info += tmp + '\n'
            else:
                raise NotImplementedError

        return info


    def forward(self, x, vars=None, bn_training=False, feature=False, svd=False):
        """
        This function can be called by finetunning, however, in finetunning, we dont wish to update
        running_mean/running_var. Thought weights/bias of bn == updated, it has been separated by fast_weights.
        Indeed, to not update running_mean/running_var, we need set update_bn_statistics=False
        but weight/bias will be updated and not dirty initial theta parameters via fast_weiths.
        :param x: [b, 1, 28, 28]
        :param vars:
        :param bn_training: set False to not update
        :return: x, loss, likelihood, kld
        """

        cat_var = False
        cat_list = []
        rep = []

        if self.args.freeze_bn and vars is not None:
            bn_vars = self.vars
        else:
            bn_vars = None

        if vars is None:
            vars = self.vars

        idx = 0
        bn_idx = 0
        fz_idx = 0

        try:
            for (name, param, extra_name) in self.config:
                if name == 'conv2d':
                    w, b = vars[idx], vars[idx + 1]
                    if svd:
                        fs = self.conv_to_linear(x, w, stride=param[4], padding=param[5])
                        x = F.conv2d(x, w, b, stride=param[4], padding=param[5])
                        assert fs.shape[0] == x.shape[0] * x.shape[2] * x.shape[3] and fs.shape[1] == x.shape[1] * \
                               w.shape[2] * w.shape[3]
                        rep.append(fs)
                    else:
                        x = F.conv2d(x, w, b, stride=param[4], padding=param[5])
                    idx += 2
                    fz_idx += 2

                elif name == 'conv2d-nbias':
                    w = vars[idx]
                    if svd:
                        fs = self.conv_to_linear(x, w, stride=param[4], padding=param[5])
                        x = F.conv2d(x, w, bias=None, stride=param[4], padding=param[5])
                        assert fs.shape[0] == x.shape[0] * x.shape[2] * x.shape[3]
                        assert fs.shape[1] == w.shape[1] * w.shape[2] * w.shape[3]
                        rep.append(fs)
                    else:
                        x = F.conv2d(x, w, bias=None, stride=param[4], padding=param[5])

                    idx += 1
                    fz_idx += 1

                elif name == 'linear':
                    if extra_name == 'cosine':
                        w = F.normalize(vars[idx])
                        x = F.normalize(x)
                        x = F.linear(x, w)
                        idx += 1
                        fz_idx += 1
                    else:
                        w, b = vars[idx], vars[idx + 1]
                        x = F.linear(x, w, b)
                        idx += 2
                        fz_idx += 2

                    if cat_var:
                        cat_list.append(x)

                elif name == 'linear-nbias':
                    if extra_name == 'cosine':
                        w = F.normalize(vars[idx])
                        x = F.normalize(x)
                        if svd:
                            rep.append(x)
                        x = F.linear(x, w)
                        idx += 1
                        fz_idx += 1
                    else:
                        w = vars[idx]
                        if svd:
                            rep.append(x)
                        x = F.linear(x, w)
                        idx += 1
                        fz_idx += 1

                    if cat_var:
                        cat_list.append(x)

                elif name == 'head-nbias':
                    w = vars[idx]
                    x = F.linear(x, w)
                    idx += 1
                    fz_idx += 1
                    if cat_var:
                        cat_list.append(x)

                elif name == 'rep':
                    if feature:
                        return x

                elif name == "cat_start":
                    cat_var = True
                    cat_list = []

                elif name == "cat":
                    cat_var = False
                    x = torch.cat(cat_list, dim=1)

                elif name == 'bn':
                    if self.training:
                        bn_training = True
                    else:
                        bn_training = (self.vars_bn is None)

                    if self.args.use_track:
                        running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx+1]
                        bn_idx += 2
                    else:
                        running_mean = None
                        running_var = None

                    if bn_vars is not None:
                        w, b = bn_vars[fz_idx], bn_vars[fz_idx + 1]
                        fz_idx += 2
                    else:
                        w, b = vars[idx], vars[idx + 1]
                        idx += 2

                    x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=bn_training)

                elif name == 'flatten':
                    x = x.view(x.size(0), -1)

                elif name == 'reshape':
                    # [b, 8] => [b, 2, 2, 2]
                    x = x.view(x.size(0), *param)
                elif name == 'relu':
                    x = F.relu(x, inplace=param[0])
                elif name == 'leakyrelu':
                    x = F.leaky_relu(x, negative_slope=param[0], inplace=param[1])
                elif name == 'tanh':
                    x = F.tanh(x)
                elif name == 'sigmoid':
                    x = torch.sigmoid(x)
                elif name == 'upsample':
                    x = F.upsample_nearest(x, scale_factor=param[0])
                elif name == 'max_pool2d':
                    x = F.max_pool2d(x, param[0], param[1], param[2])
                elif name == 'avg_pool2d':
                    x = F.avg_pool2d(x, param[0], param[1], param[2])

                else:
                    print(name)
                    raise NotImplementedError

        except:
            traceback.print_exc(file=sys.stdout)

        if svd:
            return rep
        else:
            # make sure variable == used properly
            assert idx == len(vars)
            if self.args.use_track:
                assert bn_idx == len(self.vars_bn)
            return x


    def zero_grad(self, vars=None):
        """

        :param vars:
        :return:
        """
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()


    def define_task_lr_params(self, alpha_init=1e-3):
        # Setup learning parameters
        self.alpha_lr = nn.ParameterList([])

        self.lr_name = []
        for n, p in self.named_parameters():
            self.lr_name.append(n)

        for p in self.get_params(freeze_bn=self.args.freeze_bn):
            self.alpha_lr.append(nn.Parameter(alpha_init * torch.ones(p.shape, requires_grad=True)))

    def get_params(self, freeze_bn=False):
        """
        override this function since initial parameters will return with a generator.
        :return:
        """
        if freeze_bn:
            vars = nn.ParameterList()
            for i, m in enumerate(self.vars):
                if m.ndim != 1:
                    vars.append(self.vars[i])

            return vars

        else:
            return self.vars

    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        :return:
        """
        return self.vars

    def conv_to_linear(self, x, weight, stride=1, padding=0, batchsize=None):
        if batchsize is None:
            batchsize = x.shape[0]

        if padding > 0:
            y = torch.zeros((x.shape[0], x.shape[1], x.shape[2] + 2 * padding, x.shape[3] + 2 * padding))
            y[:, :, padding: x.shape[2] + padding, padding: x.shape[2] + padding] = x
        else:
            y = x

        h = y.shape[2]
        w = y.shape[3]
        kh = weight.shape[2]
        kw = weight.shape[3]

        fs = []

        for i in range(0, h, stride):
            for j in range(0, w, stride):
                if i + kh > h or j + kw > w:
                    break
                f = y[:, :, i:i + kh, j:j + kw]
                f = f.reshape(batchsize, 1, -1)
                if i == 0 and j == 0:
                    fs = f
                else:
                    fs = torch.cat((fs, f), 1)

        fs = fs.reshape(-1, fs.shape[-1])

        return fs
