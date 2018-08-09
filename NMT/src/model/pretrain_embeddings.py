# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import os
import sys
import numpy as np
import torch


logger = getLogger()


def reload_pth_emb(path, dim):
    """
    Reload pretrained embeddings from a PyTorch binary file.
    """
    from ..data import dictionary
    sys.modules['src.dictionary'] = dictionary
    logger.info("Reloading embeddings from %s ..." % path)
    data = torch.load(path)
    vectors = data['vectors']
    logger.info("Reloaded %i embeddings." % len(vectors))
    assert vectors.size() == (len(data['dico']), dim)
    return vectors.numpy(), data['dico'].word2id


def reload_txt_emb(path, dim):
    """
    Reload pretrained embeddings from a text file.
    """
    assert os.path.isfile(path) and dim > 0
    word2id = {}
    vectors = []

    logger.info("Reloading embeddings from %s ..." % path)

    # load pretrained embeddings
    with open(path, encoding='UTF-8') as f:
        for i, line in enumerate(f):
            if i == 0:
                split = line.split()
                assert len(split) == 2
                assert dim == int(split[1])
                n_words = int(split[0])
            else:
                word, vect = line.rstrip().split(' ', 1)
                vect = np.fromstring(vect, sep=' ').astype(np.float32)
                if np.linalg.norm(vect) == 0:  # avoid to have null embeddings
                    logger.warning('Found NULL embedding for "%s" in line %i.' % (word, i))
                    vect[0] = 0.01
                assert word not in word2id
                assert vect.shape == (dim,), i
                word2id[word] = len(word2id)
                vectors.append(vect[None])
        assert len(vectors) == n_words

    vectors = np.concatenate(vectors, 0)
    assert vectors.shape == (n_words, dim)

    logger.info("Reloaded %i embeddings." % len(vectors))
    return vectors, word2id


def reload_embeddings(path, dim):
    """
    Reload pretrained embeddings.
    """
    if path.endswith('pth'):
        return reload_pth_emb(path, dim)
    else:
        return reload_txt_emb(path, dim)


def initialize_embeddings(encoder, decoder, params, data):
    """
    Initialize the model with pretrained embeddings.
    """
    if params.pretrained_emb == '':
        return

    split = params.pretrained_emb.split(',')

    if len(split) == 1:
        assert os.path.isfile(params.pretrained_emb)
        pretrained_0, word2id_0 = reload_embeddings(params.pretrained_emb, params.emb_dim)
        pretrained = [pretrained_0 for _ in range(params.n_langs)]
        word2id = [word2id_0 for _ in range(params.n_langs)]
    else:
        assert len(split) == params.n_langs
        assert not params.share_lang_emb
        assert all(os.path.isfile(x) for x in split)
        pretrained = []
        word2id = []
        for path in split:
            pretrained_i, word2id_i = reload_embeddings(path, params.emb_dim)
            pretrained.append(pretrained_i)
            word2id.append(word2id_i)

    assert not params.share_lang_emb or all(data['dico'][params.langs[i]] == data['dico'][params.langs[0]] for i in range(1, params.n_langs))

    found = [0 for _ in range(params.n_langs)]
    lower = [0 for _ in range(params.n_langs)]

    # for every language
    for i, lang in enumerate(params.langs):

        # if everything is shared, we just need to do it for the first language
        if params.share_lang_emb and i > 0:
            break

        # define dictionary / parameters to update
        dico = data['dico'][lang]
        to_update = [encoder.embeddings[i].weight.data]
        if not params.share_encdec_emb:
            to_update.append(decoder.embeddings[i].weight.data)
        if not params.share_decpro_emb and params.pretrained_out:
            to_update.append(decoder.proj[i].weight.data)

        # for every word in that language
        for word_id in range(params.n_words[i]):
            word = dico[word_id]
            if word in word2id[i]:
                found[i] += 1
                vec = torch.from_numpy(pretrained[i][word2id[i][word]]).cuda()
                for x in to_update:
                    x[word_id] = vec
            elif word.lower() in word2id[i]:
                found[i] += 1
                lower[i] += 1
                vec = torch.from_numpy(pretrained[i][word2id[i][word.lower()]]).cuda()
                for x in to_update:
                    x[word_id] = vec

    # print summary
    for i, lang in enumerate(params.langs):
        _found = found[0 if params.share_lang_emb else i]
        _lower = lower[0 if params.share_lang_emb else i]
        logger.info(
            "Initialized %i / %i word embeddings for \"%s\" (including %i "
            "after lowercasing)." % (_found, params.n_words[i], lang, _lower)
        )
