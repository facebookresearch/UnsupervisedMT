# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from .model import LSTM_PARAMS, BILSTM_PARAMS


hashs = {}


def assert_equal(x, y):
    assert x.size() == y.size()
    assert (x.data - y.data).abs().sum() == 0


def hash_data(x):
    """
    Compute a hash on tensor data.
    """
    # TODO: make a better hash function (although this is good enough for embeddings)
    return (x.data.sum(), x.data.abs().sum())


def test_sharing(encoder, decoder, lm, params):
    """
    Test parameters sharing between the encoder,
    the decoder, and the language model.
    Test that frozen parameters are not being updated.
    """
    if not params.attention:  # TODO: implement this for seq2seq model
        return
    assert params.attention is True

    # frozen parameters
    if params.freeze_enc_emb:
        for i in range(params.n_langs):
            k = 'enc_emb_%i' % i
            if k in hashs:
                assert hash_data(encoder.embeddings[i].weight) == hashs[k]
            else:
                hashs[k] = hash_data(encoder.embeddings[i].weight)
    if params.freeze_dec_emb:
        for i in range(params.n_langs):
            k = 'dec_emb_%i' % i
            if k in hashs:
                assert hash_data(decoder.embeddings[i].weight) == hashs[k]
            else:
                hashs[k] = hash_data(decoder.embeddings[i].weight)

    #
    # encoder
    #
    # embedding layers
    if params.share_lang_emb:
        for i in range(1, params.n_langs):
            assert_equal(encoder.embeddings[i].weight, encoder.embeddings[0].weight)
    # LSTM layers
    if not params.transformer:
        for k in range(params.n_enc_layers):
            if params.n_enc_layers - k <= params.share_enc - 1:
                for i in range(1, params.n_langs):
                    for name in BILSTM_PARAMS:
                        assert_equal(getattr(encoder.lstm[i], name % k), getattr(encoder.lstm[0], name % k))
    # projection layers
    if not params.transformer and params.share_enc >= 1:
        for i in range(1, params.n_langs):
            assert_equal(encoder.proj[i].weight, encoder.proj[0].weight)

    #
    # decoder
    #
    # embedding layers
    if params.share_encdec_emb:
        for i in range(params.n_langs):
            assert_equal(encoder.embeddings[i].weight, decoder.embeddings[i].weight)
    elif params.share_lang_emb:
        for i in range(1, params.n_langs):
            assert_equal(decoder.embeddings[i].weight, decoder.embeddings[0].weight)
    # LSTM layers
    if not params.transformer:
        for k in range(params.n_dec_layers):
            if k + 1 <= params.share_dec:
                for i in range(1, params.n_langs):
                    for name in LSTM_PARAMS:
                        if k == 0:
                            assert_equal(getattr(decoder.lstm1[i], name % k), getattr(decoder.lstm1[0], name % k))
                        else:
                            assert_equal(getattr(decoder.lstm2[i], name % (k - 1)), getattr(decoder.lstm2[0], name % (k - 1)))
    # attention layers
    if not params.transformer and params.share_att_proj:
        for i in range(1, params.n_langs):
            assert_equal(decoder.att_proj[i].weight, decoder.att_proj[0].weight)
            assert_equal(decoder.att_proj[i].bias, decoder.att_proj[0].bias)
    # projection layers between LSTM and output embeddings
    if params.lstm_proj:
        if params.share_lstm_proj:
            for i in range(1, params.n_langs):
                assert_equal(decoder.lstm_proj_layers[i].weight, decoder.lstm_proj_layers[0].weight)
                assert_equal(decoder.lstm_proj_layers[i].bias, decoder.lstm_proj_layers[0].bias)
    # projection layers
    if params.share_decpro_emb:
        for i in range(params.n_langs):
            assert_equal(decoder.proj[i].weight, decoder.embeddings[i].weight)
        if params.share_lang_emb:
            assert params.share_output_emb
            for i in range(1, params.n_langs):
                assert_equal(decoder.proj[i].bias, decoder.proj[0].bias)
    elif params.share_output_emb:
        assert params.share_lang_emb
        for i in range(1, params.n_langs):
            assert_equal(decoder.proj[i].weight, decoder.proj[0].weight)
            assert_equal(decoder.proj[i].bias, decoder.proj[0].bias)

    #
    # language model
    #
    assert (not (lm is None) ^ (params.lm_after == params.lm_before == 0 and
                                params.lm_share_enc == params.lm_share_dec == 0 and
                                params.lm_share_emb is False and params.lm_share_proj is False))
    if lm is not None:
        assert lm.use_lm_enc or lm.use_lm_dec

        # encoder
        if lm.use_lm_enc:
            # embedding layers
            if params.lm_share_emb:
                for i in range(params.n_langs):
                    assert_equal(lm.lm_enc.embeddings[i].weight, encoder.embeddings[i].weight)
            # LSTM layers
            for k in range(params.lm_share_enc):
                for i in range(params.n_langs):
                    for name in LSTM_PARAMS:
                        assert_equal(getattr(lm.lm_enc.lstm[i], name % k), getattr(encoder.lstm[i], name % k))

        # encoder - reverse direction
        if lm.use_lm_enc_rev:
            # embedding layers
            if params.lm_share_emb:
                for i in range(params.n_langs):
                    assert_equal(lm.lm_enc_rev.embeddings[i].weight, encoder.embeddings[i].weight)
            # LSTM layers
            for k in range(params.lm_share_enc):
                for i in range(params.n_langs):
                    for name in LSTM_PARAMS:
                        _name = '%s_reverse' % name
                        assert_equal(getattr(lm.lm_enc_rev.lstm[i], name % k), getattr(encoder.lstm[i], _name % k))

        # decoder
        if lm.use_lm_dec:
            # embedding layers
            if params.lm_share_emb:
                for i in range(params.n_langs):
                    assert_equal(lm.lm_dec.embeddings[i].weight, decoder.embeddings[i].weight)
            # LSTM layers
            for k in range(params.lm_share_dec):
                for i in range(params.n_langs):
                    for name in LSTM_PARAMS:
                        if k == 0:
                            assert_equal(getattr(lm.lm_dec.lstm[i], name % k), getattr(decoder.lstm1[i], name % k))
                        else:
                            assert_equal(getattr(lm.lm_dec.lstm[i], name % k), getattr(decoder.lstm2[i], name % (k - 1)))
            # projection layers
            if params.lm_share_proj:
                for i in range(params.n_langs):
                    assert_equal(lm.lm_dec.proj[i].weight, decoder.proj[i].weight)
                    assert_equal(lm.lm_dec.proj[i].bias, decoder.proj[i].bias)
