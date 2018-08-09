# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import os
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from . import LatentState, LSTM_PARAMS, BILSTM_PARAMS
from .discriminator import Discriminator
from .lm import LM
from ..modules.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyLoss
from .pretrain_embeddings import initialize_embeddings
from ..utils import get_mask, reload_model
# from ..gumbel import gumbel_softmax


logger = getLogger()


class Encoder(nn.Module):

    ENC_ATTR = ['n_langs', 'n_words', ('share_lang_emb', False), 'emb_dim', 'hidden_dim', 'dropout', 'n_enc_layers', ('share_enc', False), 'pad_index', 'dis_input_proj']

    def __init__(self, params):
        """
        Encoder initialization.
        """
        super(Encoder, self).__init__()

        # model parameters
        self.n_langs = params.n_langs
        self.n_words = params.n_words
        self.share_lang_emb = params.share_lang_emb
        self.emb_dim = params.emb_dim
        self.hidden_dim = params.hidden_dim
        self.dropout = params.dropout
        self.n_enc_layers = params.n_enc_layers
        self.share_enc = params.share_enc
        self.pad_index = params.pad_index
        self.freeze_enc_emb = params.freeze_enc_emb
        self.max_len = params.max_len
        self.dis_input_proj = params.dis_input_proj
        assert not self.share_lang_emb or len(set(params.n_words)) == 1
        assert 0 <= self.share_enc <= self.n_enc_layers + 1

        # embedding layers
        if self.share_lang_emb:
            logger.info("Sharing encoder input embeddings")
            layer_0 = nn.Embedding(self.n_words[0], self.emb_dim, padding_idx=self.pad_index)
            nn.init.normal_(layer_0.weight, 0, 0.1)
            nn.init.constant_(layer_0.weight[self.pad_index], 0)
            embeddings = [layer_0 for _ in range(self.n_langs)]
        else:
            embeddings = []
            for n_words in self.n_words:
                layer_i = nn.Embedding(n_words, self.emb_dim, padding_idx=self.pad_index)
                nn.init.normal_(layer_i.weight, 0, 0.1)
                nn.init.constant_(layer_i.weight[self.pad_index], 0)
                embeddings.append(layer_i)
        self.embeddings = nn.ModuleList(embeddings)

        # LSTM layers / shared layers
        lstm = [
            nn.LSTM(self.emb_dim, self.hidden_dim, num_layers=self.n_enc_layers, dropout=self.dropout, bidirectional=True)
            for _ in range(self.n_langs)
        ]
        for k in range(self.n_enc_layers):
            if self.n_enc_layers - k <= self.share_enc - 1:
                logger.info("Sharing encoder bi-LSTM parameters for layer %i" % k)
                for i in range(1, self.n_langs):
                    for name in BILSTM_PARAMS:
                        setattr(lstm[i], name % k, getattr(lstm[0], name % k))
        self.lstm = nn.ModuleList(lstm)

        # projection layers
        if self.share_enc >= 1:
            logger.info("Sharing encoder projection layers")
            proj_0 = nn.Linear(2 * self.hidden_dim, self.emb_dim, bias=False)
            proj = [proj_0 for _ in range(self.n_langs)]
        else:
            proj = [nn.Linear(2 * self.hidden_dim, self.emb_dim, bias=False)
                    for _ in range(self.n_langs)]
        self.proj = nn.ModuleList(proj)

    def forward(self, x, lengths, lang_id):
        """
        Input:
            - LongTensor of size (slen, bs), word indices
            - List of length bs, containing the sentence lengths
            Sentences have to be ordered by decreasing length
        Output:
            - FloatTensor of size (slen, bs, 2 * hidden_dim),
              representing the encoded state of each sentence
        """
        assert type(lang_id) is int
        is_cuda = x.is_cuda
        sort_len = lengths.type_as(x.data).sort(0, descending=True)[1]
        sort_len_rev = sort_len.sort()[1]
        emb_layer = self.embeddings[lang_id]
        lstm_layer = self.lstm[lang_id]
        proj_layer = self.proj[lang_id]

        # embeddings
        slen, bs = x.size(0), x.size(1)
        if x.dim() == 2:
            embeddings = emb_layer(x.index_select(1, sort_len))
        else:
            assert x.dim() == 3 and x.size(2) == self.n_words[lang_id]
            embeddings = x.view(slen * bs, -1).mm(emb_layer.weight).view(slen, bs, self.emb_dim).index_select(1, sort_len)
        embeddings = embeddings.detach() if self.freeze_enc_emb else embeddings
        embeddings = F.dropout(embeddings, p=self.dropout, training=self.training)
        lstm_input = pack_padded_sequence(embeddings, sorted(lengths.tolist(), reverse=True))
        assert lengths.max() == slen and lengths.size(0) == bs
        assert lstm_input.data.size() == (sum(lengths), self.emb_dim)

        # LSTM
        lstm_output, (_, _) = lstm_layer(lstm_input)
        assert lstm_output.data.size() == (lengths.sum(), 2 * self.hidden_dim)

        # get a padded version of the LSTM output
        padded_output, _ = pad_packed_sequence(lstm_output)
        assert padded_output.size() == (slen, bs, 2 * self.hidden_dim)

        # project biLSTM output
        padded_output = proj_layer(padded_output.view(slen * bs, -1)).view(slen, bs, self.emb_dim)

        # re-order sentences in their original order
        padded_output = padded_output.index_select(1, sort_len_rev)

        # discriminator input
        dis_input = lstm_output.data
        if self.dis_input_proj:
            mask = get_mask(lengths, all_words=True, expand=self.emb_dim, batch_first=False, cuda=is_cuda)
            dis_input = padded_output.masked_select(mask).view(lengths.sum(), self.emb_dim)

        return LatentState(input_len=lengths, dec_input=padded_output, dis_input=dis_input)


class Decoder(nn.Module):

    DEC_ATTR = ['n_langs', 'n_words', ('share_lang_emb', False), ('share_encdec_emb', False), ('share_decpro_emb', False), ('share_att_proj', False), ('share_dec', False), 'emb_dim', 'hidden_dim', 'dropout', 'n_dec_layers', 'input_feeding', 'eos_index', 'pad_index', 'bos_index']

    def __init__(self, params, encoder):
        """
        Decoder initialization.
        """
        super(Decoder, self).__init__()

        # model parameters
        self.n_langs = params.n_langs
        self.n_words = params.n_words
        self.share_lang_emb = params.share_lang_emb
        self.share_encdec_emb = params.share_encdec_emb
        self.share_decpro_emb = params.share_decpro_emb
        self.share_output_emb = params.share_output_emb
        self.share_lstm_proj = params.share_lstm_proj
        self.share_att_proj = params.share_att_proj
        self.share_dec = params.share_dec
        self.emb_dim = params.emb_dim
        self.hidden_dim = params.hidden_dim
        self.lstm_proj = params.lstm_proj
        self.dropout = params.dropout
        self.n_dec_layers = params.n_dec_layers
        self.input_feeding = params.input_feeding
        self.freeze_dec_emb = params.freeze_dec_emb
        assert not self.share_lang_emb or len(set(params.n_words)) == 1
        assert not self.share_decpro_emb or self.lstm_proj or self.emb_dim == self.hidden_dim
        assert 0 <= self.share_dec <= self.n_dec_layers
        assert self.n_dec_layers > 1 or self.n_dec_layers == 1 and self.input_feeding

        # indexes
        self.eos_index = params.eos_index
        self.pad_index = params.pad_index
        self.bos_index = params.bos_index

        # words allowed for generation
        self.vocab_mask_neg = params.vocab_mask_neg if len(params.vocab) > 0 else None

        # embedding layers
        if self.share_encdec_emb:
            logger.info("Sharing encoder and decoder input embeddings")
            embeddings = encoder.embeddings
        else:
            if self.share_lang_emb:
                logger.info("Sharing decoder input embeddings")
                layer_0 = nn.Embedding(self.n_words[0], self.emb_dim, padding_idx=self.pad_index)
                nn.init.normal_(layer_0.weight, 0, 0.1)
                nn.init.constant_(layer_0.weight[self.pad_index], 0)
                embeddings = [layer_0 for _ in range(self.n_langs)]
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
        self.lstm1_input_size = self.emb_dim + (self.emb_dim if self.input_feeding else 0)
        self.lstm2_input_size = self.hidden_dim + (0 if self.input_feeding else self.emb_dim)
        lstm1 = [
            nn.LSTM(self.lstm1_input_size, self.hidden_dim, num_layers=1, dropout=self.dropout, bias=True)
            for _ in range(self.n_langs)
        ]
        if self.n_dec_layers > 1:
            lstm2 = [
                nn.LSTM(self.lstm2_input_size, self.hidden_dim, num_layers=self.n_dec_layers - 1, dropout=self.dropout, bias=True)
                for _ in range(self.n_langs)
            ]
        else:
            lstm2 = None
        for k in range(self.n_dec_layers):
            if k + 1 <= self.share_dec:
                logger.info("Sharing decoder LSTM parameters for layer %i" % k)
                for i in range(1, self.n_langs):
                    for name in LSTM_PARAMS:
                        if k == 0:
                            setattr(lstm1[i], name % k, getattr(lstm1[0], name % k))
                        else:
                            setattr(lstm2[i], name % (k - 1), getattr(lstm2[0], name % (k - 1)))
        self.lstm1 = nn.ModuleList(lstm1)
        self.lstm2 = nn.ModuleList(lstm2)

        # attention layers
        if self.share_att_proj:
            logger.info("Sharing decoder attention projection layers")
            att_proj_0 = nn.Linear(self.hidden_dim, self.emb_dim)
            att_proj = [att_proj_0 for _ in range(self.n_langs)]
        else:
            att_proj = [nn.Linear(self.hidden_dim, self.emb_dim) for _ in range(self.n_langs)]
        self.att_proj = nn.ModuleList(att_proj)

        # projection layers between LSTM and output embeddings
        if self.lstm_proj:
            lstm_proj_layers = [nn.Linear(self.hidden_dim, self.emb_dim) for _ in range(self.n_langs)]
            if self.share_lstm_proj:
                logger.info("Sharing decoder post-LSTM projection layers")
                for i in range(1, self.n_langs):
                    lstm_proj_layers[i].weight = lstm_proj_layers[0].weight
                    lstm_proj_layers[i].bias = lstm_proj_layers[0].bias
            self.lstm_proj_layers = nn.ModuleList(lstm_proj_layers)
            proj_output_dim = self.emb_dim
        else:
            self.lstm_proj_layers = [None for _ in range(self.n_langs)]
            proj_output_dim = self.hidden_dim

        # projection layers
        proj = [nn.Linear(proj_output_dim, n_words) for n_words in self.n_words]
        if self.share_decpro_emb:
            logger.info("Sharing input embeddings and projection matrix in the decoder")
            for i in range(self.n_langs):
                proj[i].weight = self.embeddings[i].weight
            if self.share_lang_emb:
                assert self.share_output_emb
                logger.info("Sharing decoder projection matrices")
                for i in range(1, self.n_langs):
                    proj[i].bias = proj[0].bias
        elif self.share_output_emb:
            assert self.share_lang_emb
            logger.info("Sharing decoder projection matrices")
            for i in range(1, self.n_langs):
                proj[i].weight = proj[0].weight
                proj[i].bias = proj[0].bias
        self.proj = nn.ModuleList(proj)

        self.log_sm = torch.nn.LogSoftmax()

    def get_attention(self, latent, h_t, y_t, mask, lang_id):
        """
        Compute the attention vector for a single decoder step.
        """
        att_proj = self.att_proj[lang_id]
        proj_hidden = att_proj(h_t)                                                   # (bs, emb_dim)
        proj_hidden = (proj_hidden + y_t).unsqueeze(0)                                # (1, bs, emb_dim)

        att_weights = latent * proj_hidden.expand_as(latent)                          # (xlen, bs, emb_dim)
        att_weights = att_weights.sum(2)                                              # (xlen, bs)
        att_weights = att_weights.masked_fill(mask, -1e30)                            # (xlen, bs)
        att_weights = F.softmax(att_weights.transpose(0, 1), dim=-1).transpose(0, 1)  # (xlen, bs)

        att_vectors = latent * att_weights.unsqueeze(2).expand_as(latent)             # (xlen, bs, emb_dim)
        att_vectors = att_vectors.sum(0, keepdim=True)                                # (1, bs, emb_dim)

        # print " ".join("%.4f" % x for x in att_weights.data.cpu().numpy()[:, 0])
        assert att_vectors.size() == (1, proj_hidden.size(1), self.emb_dim)
        return att_vectors

    def get_full_attention(self, latent, h, y, mask, lang_id):
        """
        Compute the attention vectors for all decoder steps.
        """
        latent = latent.transpose(0, 1)
        h = h.transpose(0, 1).contiguous()
        y = y.transpose(0, 1)

        bs = latent.size(0)
        xlen = latent.size(1)
        ylen = y.size(1)

        att_proj = self.att_proj[lang_id]
        proj_hidden = att_proj(h.view(bs * ylen, self.hidden_dim))                   # (bs * ylen, emb_dim)
        proj_hidden = proj_hidden.view(bs, ylen, self.emb_dim)                       # (bs, ylen, emb_dim)
        proj_hidden = (proj_hidden + y)                                              # (bs, ylen, emb_dim)

        att_weights = proj_hidden.bmm(latent.transpose(1, 2))                        # (bs, ylen, xlen)
        att_weights = att_weights.masked_fill(mask, -1e30)                           # (bs, ylen, xlen)
        att_weights = F.softmax(att_weights.view(bs * ylen, xlen), dim=-1)           # (bs * ylen, xlen)
        att_weights = att_weights.view(bs, ylen, xlen)                               # (bs, ylen, xlen)

        att_vectors = latent.unsqueeze(1).expand(bs, ylen, xlen, self.emb_dim)       # (bs, ylen, xlen, emb_dim)
        att_vectors = att_vectors * att_weights.unsqueeze(3).expand_as(att_vectors)  # (bs, ylen, xlen, emb_dim)
        att_vectors = att_vectors.sum(2)                                             # (bs, ylen, emb_dim)

        assert att_vectors.size() == (bs, ylen, self.emb_dim)
        return att_vectors.transpose(0, 1)

    def forward(self, encoded, y, lang_id, one_hot=False):
        """
        Input:
            - LongTensor of size (slen, bs), word indices
              or
              LongTensor of size (slen, bs, n_words), one-hot word embeddings
            - LongTensor of size (bs,), sentence lengths
            - FloatTensor of size (bs, hidden_dim), latent
              state representing sentences
        Output:
            - FloatTensor of size (slen, bs, n_words),
              representing the score of each word in each sentence
        """
        latent = encoded.dec_input
        x_len = encoded.input_len
        is_cuda = latent.is_cuda

        # check inputs
        assert type(lang_id) is int
        assert x_len.size(0) == y.size(1)
        assert latent.size() == (x_len.max(), x_len.size(0), self.emb_dim)

        # source / target
        n_words = self.n_words[lang_id]
        emb_layer = self.embeddings[lang_id]
        lstm_layer1 = self.lstm1[lang_id]
        lstm_layer2 = self.lstm2[lang_id]
        lstm_proj_layer = self.lstm_proj_layers[lang_id]
        proj_layer = self.proj[lang_id]

        # embeddings
        if one_hot:
            y_len, bs, _ = y.size()
            embeddings = y.view(y_len * bs, n_words).mm(emb_layer.weight)
            embeddings = embeddings.view(y_len, bs, self.emb_dim)
        else:
            y_len, bs = y.size()
            embeddings = emb_layer(y)
            embeddings = embeddings.detach() if self.freeze_dec_emb else embeddings
        embeddings = F.dropout(embeddings, p=self.dropout, training=self.training)

        if self.input_feeding:
            mask = get_mask(x_len, True, cuda=is_cuda) == 0  # attention mask
            h_c = None
            hidden_states = [latent.data.new(1, bs, self.hidden_dim).zero_()]
            attention_states = []

            for i in range(y_len):
                # attention layer
                attention = self.get_attention(latent, hidden_states[-1][0], embeddings[i], mask, lang_id)
                attention_states.append(attention)

                # lstm step
                lstm_input = embeddings[i:i + 1]
                lstm_input = torch.cat([lstm_input, attention], 2)
                h_t, h_c = lstm_layer1(lstm_input, h_c)
                assert h_t.size() == (1, bs, self.hidden_dim)
                hidden_states.append(h_t)

            # first layer LSTM output
            lstm_output = torch.cat(hidden_states[1:], 0)
            assert lstm_output.size() == (y_len, bs, self.hidden_dim)

            # lstm (layers > 1)
            if self.n_dec_layers > 1:
                lstm_output = F.dropout(lstm_output, p=self.dropout, training=self.training)
                lstm_output, (_, _) = lstm_layer2(lstm_output)
                assert lstm_output.size() == (y_len, bs, self.hidden_dim)

        else:
            # first LSTM layer
            lstm_output, (_, _) = lstm_layer1(embeddings)
            assert lstm_output.size() == (y_len, bs, self.hidden_dim)

            # attention layer
            mask = get_mask(x_len, True, expand=int(y_len), batch_first=True, cuda=is_cuda).transpose(1, 2) == 0
            att_input = torch.cat([latent.data.new(1, bs, self.hidden_dim).zero_(), lstm_output[:-1]], 0)
            attention = self.get_full_attention(latent, att_input, embeddings, mask, lang_id)
            assert attention.size() == (y_len, bs, self.emb_dim)

            # > 1 LSTM layers
            lstm_output = F.dropout(lstm_output, p=self.dropout, training=self.training)
            lstm_output = torch.cat([lstm_output, attention], 2)
            lstm_output, (_, _) = lstm_layer2(lstm_output)
            assert lstm_output.size() == (y_len, bs, self.hidden_dim)

        # word scores
        output = F.dropout(lstm_output, p=self.dropout, training=self.training).view(-1, self.hidden_dim)
        if lstm_proj_layer is not None:
            output = F.relu(lstm_proj_layer(output))
        scores = proj_layer(output)
        return scores.view(y_len, bs, n_words)

    def generate(self, encoded, lang_id, max_len=200, sample=False, temperature=None):
        """
        Generate a sentence from a given initial state.
        Input:
            - FloatTensor of size (batch_size, hidden_dim) representing
              sentences encoded in the latent space
        Output:
            - LongTensor of size (seq_len, batch_size), word indices
            - LongTensor of size (batch_size,), sentence x_len
        """
        latent = encoded.dec_input
        x_len = encoded.input_len
        is_cuda = latent.is_cuda
        one_hot = None  # [] if temperature is not None else None

        # check inputs
        assert type(lang_id) is int
        assert latent.size() == (x_len.max(), x_len.size(0), self.emb_dim)
        assert (sample is True) ^ (temperature is None)

        # source / target
        n_words = self.n_words[lang_id]
        emb_layer = self.embeddings[lang_id]
        lstm_layer1 = self.lstm1[lang_id]
        lstm_layer2 = self.lstm2[lang_id]
        lstm_proj_layer = self.lstm_proj_layers[lang_id]
        proj_layer = self.proj[lang_id]

        # initialize generated sentences batch
        slen, bs = latent.size(0), latent.size(1)
        assert x_len.max() == slen and x_len.size(0) == bs
        cur_len = 1
        decoded = torch.LongTensor(max_len, bs).fill_(self.pad_index)
        decoded = decoded.cuda() if is_cuda else decoded
        decoded[0] = self.bos_index[lang_id]

        # compute attention
        mask = get_mask(x_len, True, cuda=is_cuda) == 0
        h_c_1, h_c_2 = None, None
        hidden_states = [latent.data.new(1, bs, self.hidden_dim).zero_()]

        while cur_len < max_len:
            # previous word embeddings
            embeddings = emb_layer(decoded[cur_len - 1])
            embeddings = F.dropout(embeddings, p=self.dropout, training=self.training)

            # attention layer
            attention = self.get_attention(latent, hidden_states[-1][0], embeddings, mask, lang_id)

            # lstm step
            lstm_input = embeddings.unsqueeze(0)
            if self.input_feeding:
                lstm_input = torch.cat([lstm_input, attention], 2)
            lstm_output, h_c_1 = lstm_layer1(lstm_input, h_c_1)
            assert lstm_output.size() == (1, bs, self.hidden_dim)
            hidden_states.append(lstm_output)

            # lstm (layers > 1)
            if self.n_dec_layers > 1:
                lstm_output = F.dropout(lstm_output, p=self.dropout, training=self.training)
                if not self.input_feeding:
                    lstm_output = torch.cat([lstm_output, attention], 2)
                lstm_output, h_c_2 = lstm_layer2(lstm_output, h_c_2)
                assert lstm_output.size() == (1, bs, self.hidden_dim)

            # word scores
            output = F.dropout(lstm_output, p=self.dropout, training=self.training).view(-1, self.hidden_dim)
            if lstm_proj_layer is not None:
                output = F.relu(lstm_proj_layer(output))
            scores = proj_layer(output).view(bs, n_words)
            scores = scores.data

            # do no sample words not in the language vocabulary
            if self.vocab_mask_neg is not None:
                scores.index_fill_(1, self.vocab_mask_neg[lang_id], -1e30)

            # select next words: sample (Gumbel Softmax) or one-hot
            if sample:
                # if temperature is not None:
                #     gumbel = gumbel_softmax(scores, temperature, hard=True)
                #     next_words = gumbel.max(1)[1]
                #     one_hot.append(gumbel)
                # else:
                next_words = torch.multinomial((scores / temperature).exp(), 1).squeeze(1)
            else:
                next_words = torch.topk(scores, 1)[1].squeeze(1)
            assert next_words.size() == (bs,)
            decoded[cur_len] = next_words
            cur_len += 1

            # stop when there is a </s> in each sentence
            if decoded.eq(self.eos_index).sum(0).ne(0).sum() == bs:
                break

        # compute the length of each generated sentence, and
        # put some padding after the end of each sentence
        lengths = torch.LongTensor(bs).fill_(cur_len)
        for i in range(bs):
            for j in range(cur_len):
                if decoded[j, i] == self.eos_index:
                    if j + 1 < max_len:
                        decoded[j + 1:, i] = self.pad_index
                    lengths[i] = j + 1
                    break
            if lengths[i] == max_len:
                decoded[-1, i] = self.eos_index

        if one_hot is not None:
            one_hot = torch.cat([x.unsqueeze(0) for x in one_hot], 0)
            assert one_hot.size() == (cur_len - 1, bs, n_words)
        return decoded[:cur_len], lengths, one_hot

    def generate_beam(self, encoded, lang_id, beam_size=20, max_len=175, sample=False, temperature=None):
        """
        Generate a sentence from a given initial state.
        Input:
            - FloatTensor of size (batch_size, hidden_dim) representing
              sentences encoded in the latent space
        Output:
            - LongTensor of size (seq_len, batch_size), word indices
            - LongTensor of size (batch_size,), sentence x_len
        """
        latent = encoded.dec_input
        x_len = encoded.input_len
        is_cuda = latent.is_cuda
        one_hot = [] if temperature is not None else None

        # check inputs
        assert type(lang_id) is int
        assert beam_size >= 1
        assert latent.size() == (x_len.max(), x_len.size(0), self.emb_dim)
        assert temperature is None or sample is True

        # source / target
        n_words = self.n_words[lang_id]
        emb_layer = self.embeddings[lang_id]
        lstm_layer1 = self.lstm1[lang_id]
        lstm_layer2 = self.lstm2[lang_id]
        proj_layer = self.proj[lang_id]

        # initialize generated sentences batch
        slen, bs = latent.size(0), latent.size(1)
        assert x_len.max() == slen and x_len.size(0) == bs
        cur_len = 1
        decoded = torch.LongTensor(max_len, bs * beam_size).fill_(self.pad_index)
        decoded = decoded.cuda() if is_cuda else decoded
        decoded[0] = self.bos_index[lang_id]

        # expand tensors for beam search
        expanded_latent = latent.unsqueeze(2).expand(slen, bs, beam_size, self.emb_dim).contiguous().view(slen, bs * beam_size, self.emb_dim)
        expanded_x_len = x_len.unsqueeze(1).expand(x_len.size(0), beam_size).contiguous().view(-1)

        # currently finished sentences / current scores in all beams
        current_hyps = [[] for _ in range(bs)]
        # current_hyp_scores = torch.FloatTensor(bs * beam_size).zero_().cuda()
        # at first step, only look at the first beam
        current_hyp_scores = latent.data.new(sum([[0] + ([-np.inf] * (beam_size - 1)) for _ in range(bs)], []))

        # compute attention
        expanded_mask = get_mask(expanded_x_len, True, cuda=is_cuda) == 0
        h_c_1, h_c_2 = None, None
        hidden_states = [latent.data.new(1, bs * beam_size, self.hidden_dim).zero_()]

        while cur_len < max_len:
            # previous word embeddings
            embeddings = emb_layer(decoded[cur_len - 1])
            embeddings = F.dropout(embeddings, p=self.dropout, training=self.training)

            # attention layer
            attention = self.get_attention(expanded_latent, hidden_states[-1][0], embeddings, expanded_mask, lang_id)

            # lstm step
            lstm_input = embeddings.unsqueeze(0)
            if self.input_feeding:
                lstm_input = torch.cat([lstm_input, attention], 2)
            lstm_output, h_c_1 = lstm_layer1(lstm_input, h_c_1)
            assert lstm_output.size() == (1, bs * beam_size, self.hidden_dim)
            hidden_states.append(lstm_output)

            # lstm (layers > 1)
            if self.n_dec_layers > 1:
                lstm_output = F.dropout(lstm_output, p=self.dropout, training=self.training)
                if not self.input_feeding:
                    lstm_output = torch.cat([lstm_output, attention], 2)
                lstm_output, h_c_2 = lstm_layer2(lstm_output, h_c_2)
                assert lstm_output.size() == (1, bs * beam_size, self.hidden_dim)

            # word scores
            lstm_output = F.dropout(lstm_output, p=self.dropout, training=self.training)
            scores = self.log_sm(proj_layer(lstm_output.view(-1, self.hidden_dim)).view(bs * beam_size, n_words))

            # beam search
            scores2 = scores.data + current_hyp_scores.unsqueeze(1).expand(bs * beam_size, n_words)
            scores2 = scores2.contiguous().view(bs, beam_size * n_words)
            best_values, best_indexes = scores2.topk(2 * beam_size, dim=1, largest=True, sorted=True)

            all_next = []  # [(current hyp value, next word, position in the total batch)]
            for sent_id in range(bs):
                if len(current_hyps[sent_id]) == beam_size:  # this sentence is done
                    all_next.extend([(0, self.pad_index, 0)] * beam_size)  # pad the batch
                    continue
                offset = sent_id * beam_size
                current_next = []
                for beam_k in range(2 * beam_size):
                    word_pos = best_indexes[sent_id, beam_k]
                    beam_id = word_pos // n_words
                    word_id = word_pos % n_words
                    assert 0 <= beam_id < beam_size and 0 <= word_id < n_words
                    if word_id == self.eos_index or cur_len + 1 == max_len:
                        current_hyps[sent_id].append((
                            decoded[:cur_len, offset + beam_id].clone(), best_values[sent_id, beam_k]
                        ))
                    else:
                        current_next.append((best_values[sent_id, beam_k], word_id, offset + beam_id))
                    if len(current_hyps[sent_id]) == beam_size:  # this sentence is done
                        current_next = [(0, self.pad_index, 0)] * beam_size  # pad the batch
                        break
                    if len(current_next) == beam_size:  # enough next hypothesis in the beam
                        break
                assert len(current_next) == beam_size
                all_next.extend(current_next)
                assert len(all_next) == beam_size * (sent_id + 1)
            # update current_hyp_scores
            # print(cur_len,  "aaaaaaaaaaaaaaaaaaaaaaaaaaa")
            assert len(all_next) == bs * beam_size, (len(all_next), bs * beam_size)
            current_hyp_scores = latent.data.new([x[0] for x in all_next])
            # print(all_next)

            # update decoded tensor, and LSTM 1 + LSTM 2 internal states
            # make this faster!!!
            slow = True
            if slow:
                _decoded = decoded.clone()
                _h_c_1 = (h_c_1[0].data.clone(), h_c_1[1].data.clone())
                _h_c_2 = (h_c_2[0].data.clone(), h_c_2[1].data.clone())
                for sent_id in range(bs):
                    for beam_k in range(beam_size):
                        k = sent_id * beam_size + beam_k
                        # print(k)
                        previous_beam_id = all_next[k][2]
                        _decoded[:, k].copy_(decoded[:, previous_beam_id].clone())
                        _decoded[cur_len, k] = all_next[k][1]
                        _h_c_1[0][0, k].copy_(_h_c_1[0][0, previous_beam_id])
                        _h_c_1[1][0, k].copy_(_h_c_1[1][0, previous_beam_id])
                        for jj in range(self.n_dec_layers - 1):
                            _h_c_2[0][jj, k].copy_(_h_c_2[0][jj, previous_beam_id])
                            _h_c_2[1][jj, k].copy_(_h_c_2[1][jj, previous_beam_id])
                decoded = _decoded
                h_c_1 = (_h_c_1[0], _h_c_1[1])
                h_c_2 = (_h_c_2[0], _h_c_2[1])
            else:
                next_ids = x_len.new([x[2] for x in all_next])
                decoded = decoded.index_select(1, next_ids)
                h_c_1 = (
                    h_c_1[0][0].index_select(0, next_ids).unsqueeze(0),
                    h_c_1[1][0].index_select(0, next_ids).unsqueeze(0)
                )
                h_c_2 = (
                    h_c_2[0][0].index_select(0, next_ids).unsqueeze(0),
                    h_c_2[1][0].index_select(0, next_ids).unsqueeze(0)
                )

            cur_len += 1

            # stop when there are `beam_size` hypothesis for each sentence
            if all(len(hyps) == beam_size for hyps in current_hyps):
                break

        def score_sent(score, sent_len):
            # return score
            return score / sent_len
            # return score / sent_len ** 0.5
            # return score / sent_len + 0.01 * sent_len

        # best hypothesis
        lengths = torch.LongTensor(bs)
        sentences = []
        for i in range(bs):
            hyps = current_hyps[i]
            assert len(hyps) == beam_size
            # best_hypo = max(hyps, key=lambda x: x[1])[0]
            # print("\n".join([" ".join([(self.src_dico if source else self.tgt_dico)[wid] for wid in hyp[0]]) for hyp in hyps]))
            best_hypo = max(hyps, key=lambda x: score_sent(x[1], len(x[0])))[0]
            lengths[i] = len(best_hypo) + 1
            sentences.append(best_hypo)
        decoded = torch.LongTensor(lengths.max(), bs).fill_(self.pad_index)
        decoded = decoded.cuda() if is_cuda else decoded
        for i, hypo in enumerate(sentences):
            decoded[:lengths[i] - 1, i] = hypo
            decoded[lengths[i] - 1, i] = self.eos_index

        # if one_hot is not None:
        #     one_hot = torch.cat([x.unsqueeze(0) for x in one_hot], 0)
        #     assert one_hot.size() == (cur_len - 1, bs, n_words)

        return decoded[:cur_len], lengths, one_hot


def build_lstm_enc_dec(params):
    logger.info("============ Building LSTM attention model - Encoder ...")
    encoder = Encoder(params)
    logger.info("")
    logger.info("============ Building LSTM attention model - Decoder ...")
    decoder = Decoder(params, encoder)
    logger.info("")
    return encoder, decoder


def build_transformer_enc_dec(params):
    from .transformer import TransformerEncoder, TransformerDecoder

    params.left_pad_source = False
    params.left_pad_target = False

    assert hasattr(params, 'dropout')
    assert hasattr(params, 'attention_dropout')
    assert hasattr(params, 'relu_dropout')

    params.encoder_embed_dim = params.emb_dim
    params.encoder_ffn_embed_dim = params.transformer_ffn_emb_dim
    params.encoder_layers = params.n_enc_layers
    assert hasattr(params, 'encoder_attention_heads')
    assert hasattr(params, 'encoder_normalize_before')

    params.decoder_embed_dim = params.emb_dim
    params.decoder_ffn_embed_dim = params.transformer_ffn_emb_dim
    params.decoder_layers = params.n_dec_layers
    assert hasattr(params, 'decoder_attention_heads')
    assert hasattr(params, 'decoder_normalize_before')

    logger.info("============ Building transformer attention model - Encoder ...")
    encoder = TransformerEncoder(params)
    logger.info("")
    logger.info("============ Building transformer attention model - Decoder ...")
    decoder = TransformerDecoder(params, encoder)
    logger.info("")
    return encoder, decoder


def build_attention_model(params, data, cuda=True):
    """
    Build a encoder / decoder, and the decoder reconstruction loss function.
    """
    # encoder / decoder / discriminator
    if params.transformer:
        encoder, decoder = build_transformer_enc_dec(params)
    else:
        encoder, decoder = build_lstm_enc_dec(params)
    if params.lambda_dis not in ["0", "-1"]:
        logger.info("============ Building attention model - Discriminator ...")
        discriminator = Discriminator(params)
        logger.info("")
    else:
        discriminator = None

    # loss function for decoder reconstruction
    loss_fn = []
    for n_words in params.n_words:
        loss_weight = torch.FloatTensor(n_words).fill_(1)
        loss_weight[params.pad_index] = 0
        if params.label_smoothing <= 0:
            loss_fn.append(nn.CrossEntropyLoss(loss_weight, size_average=True))
        else:
            loss_fn.append(LabelSmoothedCrossEntropyLoss(
                params.label_smoothing,
                params.pad_index,
                size_average=True,
                weight=loss_weight,
            ))
    decoder.loss_fn = nn.ModuleList(loss_fn)

    # language model
    if params.lambda_lm not in ["0", "-1"]:
        logger.info("============ Building attention model - Language model ...")
        lm = LM(params, encoder, decoder)
        logger.info("")
    else:
        lm = None

    # cuda - models on CPU will be synchronized and don't need to be reloaded
    if cuda:
        encoder.cuda()
        decoder.cuda()
        if len(params.vocab) > 0:
            decoder.vocab_mask_neg = [x.cuda() for x in decoder.vocab_mask_neg]
        if discriminator is not None:
            discriminator.cuda()
        if lm is not None:
            lm.cuda()

        # initialize the model with pretrained embeddings
        assert not (getattr(params, 'cpu_thread', False)) ^ (data is None)
        if data is not None:
            initialize_embeddings(encoder, decoder, params, data)

        # reload encoder / decoder / discriminator
        if params.reload_model != '':
            assert os.path.isfile(params.reload_model)
            logger.info("Reloading model from %s ..." % params.reload_model)
            reloaded = torch.load(params.reload_model)
            if params.reload_enc:
                logger.info("Reloading encoder...")
                enc = reloaded.get('enc', reloaded.get('encoder'))
                reload_model(encoder, enc, encoder.ENC_ATTR)
            if params.reload_dec:
                logger.info("Reloading decoder...")
                dec = reloaded.get('dec', reloaded.get('decoder'))
                reload_model(decoder, dec, decoder.DEC_ATTR)
            if params.reload_dis:
                assert discriminator is not None
                logger.info("Reloading discriminator...")
                dis = reloaded.get('dis', reloaded.get('discriminator'))
                reload_model(discriminator, dis, discriminator.DIS_ATTR)

    # log models
    encdec_params = set(
        p
        for module in [encoder, decoder]
        for p in module.parameters()
        if p.requires_grad
    )
    num_encdec_params = sum(p.numel() for p in encdec_params)
    logger.info("============ Model summary")
    logger.info("Number of enc+dec parameters: {}".format(num_encdec_params))
    logger.info("Encoder: {}".format(encoder))
    logger.info("Decoder: {}".format(decoder))
    logger.info("Discriminator: {}".format(discriminator))
    logger.info("LM: {}".format(lm))
    logger.info("")

    return encoder, decoder, discriminator, lm
