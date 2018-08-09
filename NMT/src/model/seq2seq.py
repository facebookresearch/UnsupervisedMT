# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import os
import torch
from torch import nn
from torch.nn import functional as F

from . import LatentState, LSTM_PARAMS
from .discriminator import Discriminator
from .lm import LM
from .pretrain_embeddings import initialize_embeddings
from ..utils import get_mask, reload_model


logger = getLogger()


def get_init_state(n_dec_layers, batch_size, hidden_dim, init_state=None):
    """
    Build an initial LSTM state, with optional non-zero first layer.
    """
    init = torch.cuda.FloatTensor(n_dec_layers, batch_size, hidden_dim).zero_()
    h_0 = init.clone()
    c_0 = init.clone()
    if init_state is not None:
        assert init_state.size() == (batch_size, hidden_dim)
        h_0[0] = init_state
    return (h_0, c_0)


class Encoder(nn.Module):

    ENC_ATTR = ['n_langs', 'n_words', ('share_lang_emb', False), 'emb_dim', 'hidden_dim', 'dropout', 'n_enc_layers', 'enc_dim', ('share_enc', False), 'proj_mode', 'pad_index']

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
        self.enc_dim = params.enc_dim
        self.share_enc = params.share_enc
        self.proj_mode = params.proj_mode
        self.pad_index = params.pad_index
        self.freeze_enc_emb = params.freeze_enc_emb
        assert not self.share_lang_emb or len(set(params.n_words)) == 1
        assert 0 <= self.share_enc <= self.n_enc_layers + int(self.proj_mode == 'proj')

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
            nn.LSTM(self.emb_dim, self.hidden_dim, num_layers=self.n_enc_layers, dropout=self.dropout)
            for _ in range(self.n_langs)
        ]
        for k in range(self.n_enc_layers):
            if self.n_enc_layers - k <= self.share_enc - int(self.proj_mode == 'proj'):
                logger.info("Sharing encoder LSTM parameters for layer %i" % k)
                for i in range(1, self.n_langs):
                    for name in LSTM_PARAMS:
                        setattr(lstm[i], name % k, getattr(lstm[0], name % k))
        self.lstm = nn.ModuleList(lstm)

        # projection layers
        if self.proj_mode == 'proj':
            if self.share_enc >= 1:
                logger.info("Sharing encoder projection layers")
                proj_0 = nn.Linear(self.hidden_dim, self.enc_dim)
                proj = [proj_0 for _ in range(self.n_langs)]
            else:
                proj = [nn.Linear(self.hidden_dim, self.enc_dim) for _ in range(self.n_langs)]
            self.proj = nn.ModuleList(proj)
        else:
            self.proj = [None for _ in range(self.n_langs)]

    def forward(self, x, lengths, lang_id):
        """
        Input:
            - LongTensor of size (slen, bs), word indices
            - LongTensor of size (bs,), sentence lengths
        Output:
            - FloatTensor of size (bs, enc_dim),
              representing the encoded state of each sentence
        """
        assert type(lang_id) is int
        is_cuda = x.is_cuda
        emb_layer = self.embeddings[lang_id]
        lstm_layer = self.lstm[lang_id]
        proj_layer = self.proj[lang_id]

        # embeddings
        slen, bs = x.size(0), x.size(1)
        if x.dim() == 2:
            embeddings = emb_layer(x)
        else:
            assert x.dim() == 3 and x.size(2) == self.n_words[lang_id]
            embeddings = x.view(slen * bs, -1).mm(emb_layer.weight).view(slen, bs, self.emb_dim)
        embeddings = embeddings.detach() if self.freeze_enc_emb else embeddings
        embeddings = F.dropout(embeddings, p=self.dropout, training=self.training)
        assert lengths.max() == slen and lengths.size(0) == bs
        assert embeddings.size() == (slen, bs, self.emb_dim)

        # LSTM
        lstm_output, (_, _) = lstm_layer(embeddings)
        assert lstm_output.size() == (slen, bs, self.hidden_dim)

        # encoded sentences representation
        if self.proj_mode == 'pool':
            latent_state = lstm_output.max(0)[0]
        else:
            # select the last state of each sentence
            mask = get_mask(lengths, False, expand=self.hidden_dim, batch_first=True, cuda=is_cuda)
            h_t = lstm_output.transpose(0, 1).masked_select(mask).view(bs, self.hidden_dim)
            if self.proj_mode == 'proj':
                latent_state = proj_layer(h_t)
            elif self.proj_mode == 'last':
                latent_state = h_t

        return LatentState(input_len=lengths, dec_input=latent_state, dis_input=latent_state)


class Decoder(nn.Module):

    DEC_ATTR = ['n_langs', 'n_words', ('share_lang_emb', False), ('share_encdec_emb', False), ('share_decpro_emb', False), ('share_dec', False), 'emb_dim', 'hidden_dim', 'dropout', 'n_dec_layers', 'enc_dim', 'init_encoded', 'eos_index', 'pad_index', 'bos_index']

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
        self.share_dec = params.share_dec
        self.emb_dim = params.emb_dim
        self.hidden_dim = params.hidden_dim
        self.lstm_proj = params.lstm_proj
        self.dropout = params.dropout
        self.n_dec_layers = params.n_dec_layers
        self.enc_dim = params.enc_dim
        self.init_encoded = params.init_encoded
        self.freeze_dec_emb = params.freeze_dec_emb
        assert not self.share_lang_emb or len(set(params.n_words)) == 1
        assert not self.share_decpro_emb or self.lstm_proj or self.emb_dim == self.hidden_dim
        assert 0 <= self.share_dec <= self.n_dec_layers
        assert self.enc_dim == self.hidden_dim or not self.init_encoded

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
        input_dim = self.emb_dim + (0 if self.init_encoded else self.enc_dim)
        lstm = [
            nn.LSTM(input_dim, self.hidden_dim, num_layers=self.n_dec_layers, dropout=self.dropout)
            for _ in range(self.n_langs)
        ]
        for k in range(self.n_dec_layers):
            if k + 1 <= self.share_dec:
                logger.info("Sharing decoder LSTM parameters for layer %i" % k)
                for i in range(1, self.n_langs):
                    for name in LSTM_PARAMS:
                        setattr(lstm[i], name % k, getattr(lstm[0], name % k))
        self.lstm = nn.ModuleList(lstm)

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
        assert type(lang_id) is int
        assert encoded.input_len.size(0) == encoded.dec_input.size(0)
        latent = encoded.dec_input
        n_words = self.n_words[lang_id]
        emb_layer = self.embeddings[lang_id]
        lstm_layer = self.lstm[lang_id]
        lstm_proj_layer = self.lstm_proj_layers[lang_id]
        proj_layer = self.proj[lang_id]

        # embeddings
        if one_hot:
            slen, bs, _ = y.size()
            embeddings = y.view(slen * bs, n_words).mm(emb_layer.weight)
            embeddings = embeddings.view(slen, bs, self.emb_dim)
        else:
            slen, bs = y.size()
            embeddings = emb_layer(y)
        embeddings = embeddings.detach() if self.freeze_dec_emb else embeddings
        embeddings = F.dropout(embeddings, p=self.dropout, training=self.training)
        assert latent.size() == (bs, self.enc_dim)
        assert embeddings.size() == (slen, bs, self.emb_dim)

        if self.init_encoded:
            init = get_init_state(self.n_dec_layers, bs, self.hidden_dim, latent)
            lstm_input = embeddings
        else:
            init = None
            encoded = latent.unsqueeze(0).expand(slen, bs, self.enc_dim)
            lstm_input = torch.cat([embeddings, encoded], 2)

        # LSTM
        lstm_output, (_, _) = lstm_layer(lstm_input, init)
        assert lstm_output.size() == (slen, bs, self.hidden_dim)

        # word scores
        output = F.dropout(lstm_output, p=self.dropout, training=self.training).view(-1, self.hidden_dim)
        if lstm_proj_layer is not None:
            output = F.relu(lstm_proj_layer(output))
        scores = proj_layer(output)
        return scores.view(slen, bs, n_words)

    def generate(self, encoded, lang_id, max_len=200, sample=False, temperature=None):
        """
        Generate a sentence from a given initial state.
        Input:
            - FloatTensor of size (batch_size, hidden_dim) representing
              sentences encoded in the latent space
        Output:
            - LongTensor of size (seq_len, batch_size), word indices
            - LongTensor of size (batch_size,), sentence lengths
        """
        assert encoded.input_len.size(0) == encoded.dec_input.size(0)
        latent = encoded.dec_input
        is_cuda = latent.is_cuda
        assert type(lang_id) is int
        assert (sample is True) ^ (temperature is None)
        one_hot = None  # [] if temperature is not None else None
        n_words = self.n_words[lang_id]
        emb_layer = self.embeddings[lang_id]
        lstm_layer = self.lstm[lang_id]
        lstm_proj_layer = self.lstm_proj_layers[lang_id]
        proj_layer = self.proj[lang_id]

        # initialize generated sentences batch
        bs = latent.size(0)
        cur_len = 1
        if self.init_encoded:
            h_c = get_init_state(self.n_dec_layers, bs, self.hidden_dim, latent)
        else:
            h_c = None
        decoded = torch.LongTensor(max_len, bs).fill_(self.pad_index)
        decoded = decoded.cuda() if is_cuda else decoded
        decoded[0] = self.bos_index[lang_id]

        # decoding
        while cur_len < max_len:
            # previous word embeddings
            embeddings = emb_layer(decoded[cur_len - 1])
            embeddings = F.dropout(embeddings, p=self.dropout, training=self.training)
            if not self.init_encoded:
                embeddings = torch.cat([embeddings, latent], 1)
            lstm_output, h_c = lstm_layer(embeddings.unsqueeze(0), h_c)
            output = F.dropout(lstm_output, p=self.dropout, training=self.training).view(bs, self.hidden_dim)
            if lstm_proj_layer is not None:
                output = F.relu(lstm_proj_layer(output))
            scores = proj_layer(output).data
            assert scores.size() == (bs, n_words)

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
                next_words = scores.max(1)[1]
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


def build_seq2seq_model(params, data, cuda=True):
    """
    Build a encoder / decoder, and the decoder reconstruction loss function.
    """
    # encoder / decoder / discriminator
    logger.info("============ Building seq2seq model - Encoder ...")
    encoder = Encoder(params)
    logger.info("")
    logger.info("============ Building seq2seq model - Decoder ...")
    decoder = Decoder(params, encoder)
    logger.info("")
    if params.lambda_dis not in ["0", "-1"]:
        logger.info("============ Building seq2seq model - Discriminator ...")
        discriminator = Discriminator(params)
        logger.info("")
    else:
        discriminator = None

    # loss function for decoder reconstruction
    loss_fn = []
    for n_words in params.n_words:
        loss_weight = torch.FloatTensor(n_words).fill_(1)
        loss_weight[params.pad_index] = 0
        loss_fn.append(nn.CrossEntropyLoss(loss_weight, size_average=True))
    decoder.loss_fn = nn.ModuleList(loss_fn)

    # language model
    if params.lambda_lm not in ["0", "-1"]:
        logger.info("============ Building seq2seq model - Language model ...")
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
                reload_model(encoder, reloaded['enc'], encoder.ENC_ATTR)
            if params.reload_dec:
                logger.info("Reloading decoder...")
                reload_model(decoder, reloaded['dec'], decoder.DEC_ATTR)
            if params.reload_dis:
                logger.info("Reloading discriminator...")
                reload_model(discriminator, reloaded['dis'], discriminator.DIS_ATTR)

    # log models
    logger.info("============ Model summary")
    logger.info("Encoder: {}".format(encoder))
    logger.info("Decoder: {}".format(decoder))
    logger.info("Discriminator: {}".format(discriminator))
    logger.info("LM: {}".format(lm))
    logger.info("")

    return encoder, decoder, discriminator, lm
