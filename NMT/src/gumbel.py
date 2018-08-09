# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
from torch import nn


def sample_gumbel(shape, eps=1e-20):
    """
    Sample from Gumbel(0, 1).
    """
    u = torch.cuda.FloatTensor(*shape).uniform_()
    return -torch.log(-torch.log(u + eps) + eps)


def gumbel_softmax_sample(log_probs, temperature):
    """
    Draw a sample from the Gumbel-Softmax distribution.
    """
    y = log_probs + sample_gumbel(log_probs.size())
    return nn.Softmax()(y / temperature)


def gumbel_softmax(log_probs, temperature, hard=False):
    """
    Sample from the Gumbel-Softmax distribution and optionally discretize.
    Input:
        log_probs: (batch_size, n_class) unnormalized log-probs
        temperature: non-negative scalar
        hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Output:
        (batch_size, n_class) sample from the Gumbel-Softmax distribution.
        If hard is True, then the returned sample will be one-hot, otherwise
        it will be a probabilitiy distribution that sums to 1 across classes.
    """
    y = gumbel_softmax_sample(log_probs, temperature)
    if hard:
        y_hard = torch.cuda.FloatTensor(y.size()).zero_()
        y_hard.scatter_(1, y.max(1)[1].data, 1)
        y = (y_hard - y).detach() + y
    return y
