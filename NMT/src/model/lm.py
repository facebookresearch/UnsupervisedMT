# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
from torch import nn
from torch.nn import functional as F

from . import LSTM_PARAMS


logger = getLogger()


class LM(nn.Module):

    def __init__(self, params, encoder, decoder):
        super(LM, self).__init__()
        self.use_lm_enc = params.lm_share_emb or params.lm_share_enc > 0
        self.use_lm_dec = params.lm_share_emb or params.lm_share_dec > 0 or params.lm_share_proj
        self.use_lm_enc_rev = self.use_lm_enc and params.attention
        self.lm_enc = SubLM(params, encoder, True, False) if self.use_lm_enc else None
        self.lm_dec = SubLM(params, decoder, False, False) if self.use_lm_dec else None
        self.lm_enc_rev = SubLM(params, encoder, True, True) if self.use_lm_enc_rev else None

    def forward(self, x, lengths, source, is_encoder, reverse):
        assert type(is_encoder) is bool and type(reverse) is bool
        assert reverse is False or self.use_lm_enc_rev
        if is_encoder:
            model = self.lm_enc_rev if reverse else self.lm_enc
        else:
            model = self.lm_dec
        return model(x, lengths, source)


class SubLM(nn.Module):

    def __init__(self, params, model, is_encoder, reverse):
        """
        Language model initialization.
        """
        super(SubLM, self).__init__()
        assert type(is_encoder) is bool and type(reverse) is bool
        assert reverse is False or is_encoder and params.attention

        # model parameters
        self.n_langs = params.n_langs
        self.n_words = params.n_words
        self.emb_dim = params.emb_dim
        self.hidden_dim = params.hidden_dim
        self.dropout = params.dropout
        self.pad_index = params.pad_index
        self.is_enc_lm = is_encoder
        s_name = "encoder" if is_encoder else "decoder"
        assert 0 <= params.lm_share_enc <= params.n_enc_layers
        assert 0 <= params.lm_share_dec <= params.n_dec_layers

        # embedding layers
        if params.lm_share_emb:
            embeddings = model.embeddings
            logger.info("Sharing language model input embeddings with the %s" % s_name)
        else:
            embeddings = []
            for n_words in self.n_words:
                layer_i = nn.Embedding(n_words, self.emb_dim, padding_idx=self.pad_index)
                nn.init.normal_(layer_i.weight, 0, 0.1)
                nn.init.constant_(layer_i.weight[self.pad_index], 0)
                embeddings.append(layer_i)
            embeddings = nn.ModuleList(embeddings)
        self.embeddings = embeddings

        # LSTM layers / shared layers
        n_rec_share = params.lm_share_enc if is_encoder else params.lm_share_dec
        lstm = [
            nn.LSTM(self.emb_dim, self.hidden_dim, num_layers=max(n_rec_share, 1), dropout=self.dropout)
            for _ in range(self.n_langs)
        ]
        for k in range(n_rec_share):
            logger.info("Sharing language model LSTM parameters for layer %i with the %s" % (k, s_name))
            for i in range(self.n_langs):
                for name in LSTM_PARAMS:
                    if is_encoder or not params.attention:
                        _name = name if reverse is False else ('%s_reverse' % name)
                        setattr(lstm[i], name % k, getattr(model.lstm[i], _name % k))
                    elif k == 0:
                        setattr(lstm[i], name % k, getattr(model.lstm1[i], name % k))
                    else:
                        setattr(lstm[i], name % k, getattr(model.lstm2[i], name % (k - 1)))
        self.lstm = nn.ModuleList(lstm)

        # projection layers
        if params.lm_share_proj and not is_encoder:
            logger.info("Sharing language model projection layer with the decoder")
            proj = model.proj
        else:
            proj = nn.ModuleList([nn.Linear(self.hidden_dim, n_words) for n_words in self.n_words])
        self.proj = proj

    def forward(self, x, lengths, lang_id):
        """
        Input:
            - LongTensor of size (slen, bs), word indices
            - LongTensor of size (bs,), sentence lengths
        Output:
            - FloatTensor of size (slen, bs, n_words),
              representing the score for each output word of being the next word
        """
        assert type(lang_id) is int
        slen, bs = x.size()
        assert lengths.max() == slen and lengths.size(0) == bs
        emb_layer = self.embeddings[lang_id]
        lstm_layer = self.lstm[lang_id]
        proj_layer = self.proj[lang_id]

        # embeddings
        embeddings = emb_layer(x)
        embeddings = F.dropout(embeddings, p=self.dropout, training=self.training)
        assert embeddings.size() == (slen, bs, self.emb_dim)

        # LSTM
        lstm_output, (_, _) = lstm_layer(embeddings)
        assert lstm_output.size() == (slen, bs, self.hidden_dim)

        # word scores
        word_scores = proj_layer(lstm_output)
        return word_scores
