# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from torch import nn


class Discriminator(nn.Module):

    DIS_ATTR = ['input_dim', 'dis_layers', 'dis_hidden_dim', 'dis_dropout']

    def __init__(self, params):
        """
        Discriminator initialization.
        """
        super(Discriminator, self).__init__()

        self.n_langs = params.n_langs
        self.input_dim = params.hidden_dim if params.attention else params.enc_dim
        self.dis_layers = params.dis_layers
        self.dis_hidden_dim = params.dis_hidden_dim
        self.dis_dropout = params.dis_dropout

        layers = []
        for i in range(self.dis_layers + 1):
            if i == 0:
                input_dim = self.input_dim
                input_dim *= (2 if params.attention and not params.dis_input_proj else 1)
            else:
                input_dim = self.dis_hidden_dim
            output_dim = self.dis_hidden_dim if i < self.dis_layers else self.n_langs
            layers.append(nn.Linear(input_dim, output_dim))
            if i < self.dis_layers:
                layers.append(nn.LeakyReLU(0.2))
                layers.append(nn.Dropout(self.dis_dropout))
        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        return self.layers(input)
