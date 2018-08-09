# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
from collections import namedtuple


LSTM_PARAMS = ['weight_ih_l%i', 'weight_hh_l%i', 'bias_ih_l%i', 'bias_hh_l%i']
BILSTM_PARAMS = LSTM_PARAMS + ['%s_reverse' % x for x in LSTM_PARAMS]

LatentState = namedtuple('LatentState', 'dec_input, dis_input, input_len')


def check_mt_model_params(params):
    """
    Check models parameters.
    """

    # shared layers
    assert 0 <= params.dropout < 1
    assert 0 <= params.share_enc <= params.n_enc_layers + int(params.attention and not params.transformer or not params.attention and params.proj_mode == 'proj')
    assert 0 <= params.share_dec <= params.n_dec_layers
    assert not params.share_decpro_emb or params.lstm_proj or getattr(params, 'transformer', False) or params.emb_dim == params.hidden_dim
    assert not params.share_output_emb or params.share_lang_emb
    assert (not (params.share_decpro_emb and params.share_lang_emb)) or params.share_output_emb
    assert not params.lstm_proj or not (params.attention and params.transformer)
    assert not params.share_lstm_proj or params.lstm_proj

    # attention model
    if params.attention:
        assert params.transformer or params.n_dec_layers > 1 or params.n_dec_layers == 1 and params.input_feeding
        assert params.transformer is False or params.emb_dim % params.encoder_attention_heads == 0
        assert params.transformer is False or params.emb_dim % params.decoder_attention_heads == 0
    # seq2seq model
    else:
        assert params.enc_dim == params.hidden_dim or not params.init_encoded
        assert params.enc_dim == params.hidden_dim or params.proj_mode == 'proj'
        assert params.proj_mode in ['proj', 'pool', 'last']

    # language model
    if params.lm_before == params.lm_after == 0:
        assert params.lm_share_enc == params.lm_share_dec == 0
        assert params.lm_share_emb is False and params.lm_share_proj is False
        assert params.lambda_lm == "0"
    else:
        assert not (params.attention and params.transformer)
        assert params.lm_share_enc <= 1 and params.lm_share_dec <= 1  # TODO: support more than one layer
        assert params.input_feeding is False or params.lm_share_dec == 0  # TODO: support input feeding mode
        assert 0 <= params.lm_share_enc <= params.n_enc_layers
        assert 0 <= params.lm_share_dec <= params.n_dec_layers
        assert (params.lm_share_enc + params.lm_share_dec > 0 or
                params.lm_share_emb or params.lm_share_proj)
        assert params.lambda_lm not in ["0", "-1"]
        assert params.lm_share_emb is False or not (params.freeze_enc_emb or params.freeze_dec_emb)

    # pretrained embeddings / freeze embeddings
    if params.pretrained_emb == '':
        assert not params.freeze_enc_emb or params.reload_enc
        assert not params.freeze_dec_emb or params.reload_dec
        assert not params.pretrained_out
    else:
        split = params.pretrained_emb.split(',')
        if len(split) == 1:
            assert os.path.isfile(params.pretrained_emb)
        else:
            assert len(split) == params.n_langs
            assert not params.share_lang_emb
            assert all(os.path.isfile(x) for x in split)
        if params.share_encdec_emb:
            assert params.freeze_enc_emb == params.freeze_dec_emb
        else:
            assert not (params.freeze_enc_emb and params.freeze_dec_emb)
        assert not (params.share_decpro_emb and params.freeze_dec_emb)
        assert not (params.share_decpro_emb and not params.pretrained_out)
        assert not params.pretrained_out or params.lstm_proj or getattr(params, 'transformer', False) or params.emb_dim == params.hidden_dim

    # discriminator parameters
    assert params.dis_layers >= 0
    assert params.dis_hidden_dim > 0
    assert 0 <= params.dis_dropout < 1
    assert params.dis_clip >= 0

    # reload MT model
    assert params.reload_model == '' or os.path.isfile(params.reload_model)
    assert not (params.reload_model != '') ^ (params.reload_enc or params.reload_dec or params.reload_dis)


def build_mt_model(params, data, cuda=True):
    """
    Build machine translation model.
    """
    if params.attention:
        from .attention import build_attention_model
        return build_attention_model(params, data, cuda=cuda)
    else:
        from .seq2seq import build_seq2seq_model
        return build_seq2seq_model(params, data, cuda=cuda)
