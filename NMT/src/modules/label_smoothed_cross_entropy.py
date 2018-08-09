# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# Adapted from fairseq-py

from torch import nn
import torch.nn.functional as F


class LabelSmoothedCrossEntropyLoss(nn.Module):

    def __init__(self, eps, padding_idx=None, size_average=True, weight=None):
        super().__init__()
        self.eps = eps
        self.padding_idx = padding_idx
        self.size_average = size_average
        self.register_buffer('weight', weight)

    def forward(self, input, target):
        lprobs = F.log_softmax(input, dim=-1)
        target = target.view(-1, 1)

        nll_loss = -lprobs.gather(dim=-1, index=target)
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
        if self.padding_idx is not None:
            non_pad_mask = target.ne(self.padding_idx)
            nll_loss = nll_loss[non_pad_mask]
            smooth_loss = smooth_loss[non_pad_mask]

        if self.size_average:
            nll_loss = nll_loss.mean()
            smooth_loss = smooth_loss.mean()
        else:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()

        eps_i = self.eps / lprobs.size(-1)
        loss = (1. - self.eps) * nll_loss + eps_i * smooth_loss

        return loss

