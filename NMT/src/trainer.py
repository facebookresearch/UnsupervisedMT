# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import time
from logging import getLogger
import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_

from .utils import reverse_sentences, clip_parameters
from .utils import get_optimizer, parse_lambda_config, update_lambdas
from .model import build_mt_model
from .multiprocessing_event_loop import MultiprocessingEventLoop
from .test import test_sharing


logger = getLogger()


class TrainerMT(MultiprocessingEventLoop):

    VALIDATION_METRICS = []

    def __init__(self, encoder, decoder, discriminator, lm, data, params):
        """
        Initialize trainer.
        """
        super().__init__(device_ids=tuple(range(params.otf_num_processes)))
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator
        self.lm = lm
        self.data = data
        self.params = params

        # initialization for on-the-fly generation/training
        if len(params.pivo_directions) > 0:
            self.otf_start_multiprocessing()

        # define encoder parameters (the ones shared with the
        # decoder are optimized by the decoder optimizer)
        enc_params = list(encoder.parameters())
        for i in range(params.n_langs):
            if params.share_lang_emb and i > 0:
                break
            assert enc_params[i].size() == (params.n_words[i], params.emb_dim)
        if self.params.share_encdec_emb:
            to_ignore = 1 if params.share_lang_emb else params.n_langs
            enc_params = enc_params[to_ignore:]

        # optimizers
        if params.dec_optimizer == 'enc_optimizer':
            params.dec_optimizer = params.enc_optimizer
        self.enc_optimizer = get_optimizer(enc_params, params.enc_optimizer) if len(enc_params) > 0 else None
        self.dec_optimizer = get_optimizer(decoder.parameters(), params.dec_optimizer)
        self.dis_optimizer = get_optimizer(discriminator.parameters(), params.dis_optimizer) if discriminator is not None else None
        self.lm_optimizer = get_optimizer(lm.parameters(), params.enc_optimizer) if lm is not None else None

        # models / optimizers
        self.model_opt = {
            'enc': (self.encoder, self.enc_optimizer),
            'dec': (self.decoder, self.dec_optimizer),
            'dis': (self.discriminator, self.dis_optimizer),
            'lm': (self.lm, self.lm_optimizer),
        }

        # define validation metrics / stopping criterion used for early stopping
        logger.info("Stopping criterion: %s" % params.stopping_criterion)
        if params.stopping_criterion == '':
            for lang1, lang2 in self.data['para'].keys():
                for data_type in ['valid', 'test']:
                    self.VALIDATION_METRICS.append('bleu_%s_%s_%s' % (lang1, lang2, data_type))
            for lang1, lang2, lang3 in self.params.pivo_directions:
                if lang1 == lang3:
                    continue
                for data_type in ['valid', 'test']:
                    self.VALIDATION_METRICS.append('bleu_%s_%s_%s_%s' % (lang1, lang2, lang3, data_type))
            self.stopping_criterion = None
            self.best_stopping_criterion = None
        else:
            split = params.stopping_criterion.split(',')
            assert len(split) == 2 and split[1].isdigit()
            self.decrease_counts_max = int(split[1])
            self.decrease_counts = 0
            self.stopping_criterion = split[0]
            self.best_stopping_criterion = -1e12
            assert len(self.VALIDATION_METRICS) == 0
            self.VALIDATION_METRICS.append(self.stopping_criterion)

        # training variables
        self.best_metrics = {metric: -1e12 for metric in self.VALIDATION_METRICS}
        self.epoch = 0
        self.n_total_iter = 0
        self.freeze_enc_emb = self.params.freeze_enc_emb
        self.freeze_dec_emb = self.params.freeze_dec_emb

        # training statistics
        self.n_iter = 0
        self.n_sentences = 0
        self.stats = {
            'dis_costs': [],
            'processed_s': 0,
            'processed_w': 0,
        }
        for lang in params.mono_directions:
            self.stats['xe_costs_%s_%s' % (lang, lang)] = []
        for lang1, lang2 in params.para_directions:
            self.stats['xe_costs_%s_%s' % (lang1, lang2)] = []
        for lang1, lang2 in params.back_directions:
            self.stats['xe_costs_bt_%s_%s' % (lang1, lang2)] = []
        for lang1, lang2, lang3 in params.pivo_directions:
            self.stats['xe_costs_%s_%s_%s' % (lang1, lang2, lang3)] = []
        for lang in params.langs:
            self.stats['lme_costs_%s' % lang] = []
            self.stats['lmd_costs_%s' % lang] = []
            self.stats['lmer_costs_%s' % lang] = []
            self.stats['enc_norms_%s' % lang] = []
        self.last_time = time.time()
        if len(params.pivo_directions) > 0:
            self.gen_time = 0

        # data iterators
        self.iterators = {}

        # initialize BPE subwords
        self.init_bpe()

        # initialize lambda coefficients and their configurations
        parse_lambda_config(params, 'lambda_xe_mono')
        parse_lambda_config(params, 'lambda_xe_para')
        parse_lambda_config(params, 'lambda_xe_back')
        parse_lambda_config(params, 'lambda_xe_otfd')
        parse_lambda_config(params, 'lambda_xe_otfa')
        parse_lambda_config(params, 'lambda_dis')
        parse_lambda_config(params, 'lambda_lm')

    def init_bpe(self):
        """
        Index BPE words.
        """
        self.bpe_end = []
        for lang in self.params.langs:
            dico = self.data['dico'][lang]
            self.bpe_end.append(np.array([not dico[i].endswith('@@') for i in range(len(dico))]))

    def get_iterator(self, iter_name, lang1, lang2, back):
        """
        Create a new iterator for a dataset.
        """
        assert back is False or lang2 is not None
        key = ','.join([x for x in [iter_name, lang1, lang2] if x is not None]) + ('_back' if back else '')
        logger.info("Creating new training %s iterator ..." % key)
        if lang2 is None:
            dataset = self.data['mono'][lang1]['train']
        elif back:
            dataset = self.data['back'][(lang1, lang2)]
        else:
            k = (lang1, lang2) if lang1 < lang2 else (lang2, lang1)
            dataset = self.data['para'][k]['train']
        iterator = dataset.get_iterator(shuffle=True, group_by_size=self.params.group_by_size)()
        self.iterators[key] = iterator
        return iterator

    def get_batch(self, iter_name, lang1, lang2, back=False):
        """
        Return a batch of sentences from a dataset.
        """
        assert back is False or lang2 is not None
        assert lang1 in self.params.langs
        assert lang2 is None or lang2 in self.params.langs
        key = ','.join([x for x in [iter_name, lang1, lang2] if x is not None]) + ('_back' if back else '')
        iterator = self.iterators.get(key, None)
        if iterator is None:
            iterator = self.get_iterator(iter_name, lang1, lang2, back)
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = self.get_iterator(iter_name, lang1, lang2, back)
            batch = next(iterator)
        return batch if (lang2 is None or lang1 < lang2 or back) else batch[::-1]

    def word_shuffle(self, x, l, lang_id):
        """
        Randomly shuffle input words.
        """
        if self.params.word_shuffle == 0:
            return x, l

        # define noise word scores
        noise = np.random.uniform(0, self.params.word_shuffle, size=(x.size(0) - 1, x.size(1)))
        noise[0] = -1  # do not move start sentence symbol

        # be sure to shuffle entire words
        bpe_end = self.bpe_end[lang_id][x]
        word_idx = bpe_end[::-1].cumsum(0)[::-1]
        word_idx = word_idx.max(0)[None, :] - word_idx

        assert self.params.word_shuffle > 1
        x2 = x.clone()
        for i in range(l.size(0)):
            # generate a random permutation
            scores = word_idx[:l[i] - 1, i] + noise[word_idx[:l[i] - 1, i], i]
            scores += 1e-6 * np.arange(l[i] - 1)  # ensure no reordering inside a word
            permutation = scores.argsort()
            # shuffle words
            x2[:l[i] - 1, i].copy_(x2[:l[i] - 1, i][torch.from_numpy(permutation)])
        return x2, l

    def word_dropout(self, x, l, lang_id):
        """
        Randomly drop input words.
        """
        if self.params.word_dropout == 0:
            return x, l
        assert 0 < self.params.word_dropout < 1

        # define words to drop
        bos_index = self.params.bos_index[lang_id]
        assert (x[0] == bos_index).sum() == l.size(0)
        keep = np.random.rand(x.size(0) - 1, x.size(1)) >= self.params.word_dropout
        keep[0] = 1  # do not drop the start sentence symbol

        # be sure to drop entire words
        bpe_end = self.bpe_end[lang_id][x]
        word_idx = bpe_end[::-1].cumsum(0)[::-1]
        word_idx = word_idx.max(0)[None, :] - word_idx

        sentences = []
        lengths = []
        for i in range(l.size(0)):
            assert x[l[i] - 1, i] == self.params.eos_index
            words = x[:l[i] - 1, i].tolist()
            # randomly drop words from the input
            new_s = [w for j, w in enumerate(words) if keep[word_idx[j, i], i]]
            # we need to have at least one word in the sentence (more than the start / end sentence symbols)
            if len(new_s) == 1:
                new_s.append(words[np.random.randint(1, len(words))])
            new_s.append(self.params.eos_index)
            assert len(new_s) >= 3 and new_s[0] == bos_index and new_s[-1] == self.params.eos_index
            sentences.append(new_s)
            lengths.append(len(new_s))
        # re-construct input
        l2 = torch.LongTensor(lengths)
        x2 = torch.LongTensor(l2.max(), l2.size(0)).fill_(self.params.pad_index)
        for i in range(l2.size(0)):
            x2[:l2[i], i].copy_(torch.LongTensor(sentences[i]))
        return x2, l2

    def word_blank(self, x, l, lang_id):
        """
        Randomly blank input words.
        """
        if self.params.word_blank == 0:
            return x, l
        assert 0 < self.params.word_blank < 1

        # define words to blank
        bos_index = self.params.bos_index[lang_id]
        assert (x[0] == bos_index).sum() == l.size(0)
        keep = np.random.rand(x.size(0) - 1, x.size(1)) >= self.params.word_blank
        keep[0] = 1  # do not blank the start sentence symbol

        # be sure to blank entire words
        bpe_end = self.bpe_end[lang_id][x]
        word_idx = bpe_end[::-1].cumsum(0)[::-1]
        word_idx = word_idx.max(0)[None, :] - word_idx

        sentences = []
        for i in range(l.size(0)):
            assert x[l[i] - 1, i] == self.params.eos_index
            words = x[:l[i] - 1, i].tolist()
            # randomly blank words from the input
            new_s = [w if keep[word_idx[j, i], i] else self.params.blank_index for j, w in enumerate(words)]
            new_s.append(self.params.eos_index)
            assert len(new_s) == l[i] and new_s[0] == bos_index and new_s[-1] == self.params.eos_index
            sentences.append(new_s)
        # re-construct input
        x2 = torch.LongTensor(l.max(), l.size(0)).fill_(self.params.pad_index)
        for i in range(l.size(0)):
            x2[:l[i], i].copy_(torch.LongTensor(sentences[i]))
        return x2, l

    def add_noise(self, words, lengths, lang_id):
        """
        Add noise to the encoder input.
        """
        words, lengths = self.word_shuffle(words, lengths, lang_id)
        words, lengths = self.word_dropout(words, lengths, lang_id)
        words, lengths = self.word_blank(words, lengths, lang_id)
        return words, lengths

    def zero_grad(self, models):
        """
        Zero gradients.
        """
        if type(models) is not list:
            models = [models]
        models = [self.model_opt[name] for name in models]
        for _, optimizer in models:
            if optimizer is not None:
                optimizer.zero_grad()

    def update_params(self, models):
        """
        Update parameters.
        """
        if type(models) is not list:
            models = [models]
        # don't update encoder when it's frozen
        models = [self.model_opt[name] for name in models]
        # clip gradients
        for model, _ in models:
            clip_grad_norm_(model.parameters(), self.params.clip_grad_norm)

        # optimizer
        for _, optimizer in models:
            if optimizer is not None:
                optimizer.step()

    def get_lrs(self, models):
        """
        Get current optimizer learning rates.
        """
        if type(models) is not list:
            models = [models]
        lrs = {}
        for name in models:
            optimizer = self.model_opt[name][1]
            if optimizer is not None:
                lrs[name] = optimizer.param_groups[0]['lr']
        return lrs

    def discriminator_step(self):
        """
        Train the discriminator on the latent space.
        """
        self.encoder.eval()
        self.decoder.eval()
        self.discriminator.train()

        # train on monolingual data only
        if self.params.n_mono == 0:
            raise Exception("No data to train the discriminator!")

        # batch / encode
        encoded = []
        for lang_id, lang in enumerate(self.params.langs):
            sent1, len1 = self.get_batch('dis', lang, None)
            with torch.no_grad():
                encoded.append(self.encoder(sent1.cuda(), len1, lang_id))

        # discriminator
        dis_inputs = [x.dis_input.view(-1, x.dis_input.size(-1)) for x in encoded]
        ntokens = [dis_input.size(0) for dis_input in dis_inputs]
        encoded = torch.cat(dis_inputs, 0)
        predictions = self.discriminator(encoded.data)

        # loss
        self.dis_target = torch.cat([torch.zeros(sz).fill_(i) for i, sz in enumerate(ntokens)])
        self.dis_target = self.dis_target.contiguous().long().cuda()
        y = self.dis_target

        loss = F.cross_entropy(predictions, y)
        self.stats['dis_costs'].append(loss.item())

        # optimizer
        self.zero_grad('dis')
        loss.backward()
        self.update_params('dis')
        clip_parameters(self.discriminator, self.params.dis_clip)

    def lm_step(self, lang):
        """
        Language model training.
        """
        assert self.params.lambda_lm > 0
        assert lang in self.params.langs
        assert self.lm.use_lm_enc or self.lm.use_lm_dec
        lang_id = self.params.lang2id[lang]
        self.lm.train()

        loss_fn = self.decoder.loss_fn[lang_id]
        n_words = self.params.n_words[lang_id]

        # batch
        sent1, len1 = self.get_batch('lm', lang, None)
        sent1 = sent1.cuda()
        if self.lm.use_lm_enc_rev:
            sent1_rev = reverse_sentences(sent1, len1)

        # forward
        if self.lm.use_lm_enc:
            scores_enc = self.lm(sent1[:-1], len1 - 1, lang_id, True, False)
        if self.lm.use_lm_dec:
            scores_dec = self.lm(sent1[:-1], len1 - 1, lang_id, False, False)
        if self.lm.use_lm_enc_rev:
            scores_enc_rev = self.lm(sent1_rev[:-1], len1 - 1, lang_id, True, True)

        # loss
        loss = 0
        if self.lm.use_lm_enc:
            loss_enc = loss_fn(scores_enc.view(-1, n_words), sent1[1:].view(-1))
            self.stats['lme_costs_%s' % lang].append(loss_enc.item())
            loss += loss_enc
        if self.lm.use_lm_dec:
            loss_dec = loss_fn(scores_dec.view(-1, n_words), sent1[1:].view(-1))
            self.stats['lmd_costs_%s' % lang].append(loss_dec.item())
            loss += loss_dec
        if self.lm.use_lm_enc_rev:
            loss_enc_rev = loss_fn(scores_enc_rev.view(-1, n_words), sent1_rev[1:].view(-1))
            self.stats['lmer_costs_%s' % lang].append(loss_enc_rev.item())
            loss += loss_enc_rev
        loss = self.params.lambda_lm * loss

        # check NaN
        if (loss != loss).data.any():
            logger.error("NaN detected")
            exit()

        # optimizer
        self.zero_grad(['lm'])
        loss.backward()
        self.update_params(['lm'])

        # number of processed sentences / words
        self.stats['processed_s'] += len1.size(0)
        self.stats['processed_w'] += len1.sum()

    def enc_dec_step(self, lang1, lang2, lambda_xe, back=False):
        """
        Source / target autoencoder training (parallel data):
            - encoders / decoders training on cross-entropy
            - encoders training on discriminator feedback
            - encoders training on L2 loss (seq2seq only, not for attention)
        """
        params = self.params
        assert lang1 in params.langs and lang2 in params.langs
        lang1_id = params.lang2id[lang1]
        lang2_id = params.lang2id[lang2]
        loss_fn = self.decoder.loss_fn[lang2_id]
        n_words = params.n_words[lang2_id]
        self.encoder.train()
        self.decoder.train()
        if self.discriminator is not None:
            self.discriminator.eval()

        # batch
        if back:
            (sent1, len1), (sent2, len2) = self.get_batch('encdec', lang1, lang2, back=True)
        elif lang1 == lang2:
            sent1, len1 = self.get_batch('encdec', lang1, None)
            sent2, len2 = sent1, len1
        else:
            (sent1, len1), (sent2, len2) = self.get_batch('encdec', lang1, lang2)

        # prepare the encoder / decoder inputs
        if lang1 == lang2:
            sent1, len1 = self.add_noise(sent1, len1, lang1_id)
        sent1, sent2 = sent1.cuda(), sent2.cuda()

        # encoded states
        encoded = self.encoder(sent1, len1, lang1_id)
        self.stats['enc_norms_%s' % lang1].append(encoded.dis_input.data.norm(2, 1).mean().item())

        # cross-entropy scores / loss
        scores = self.decoder(encoded, sent2[:-1], lang2_id)
        xe_loss = loss_fn(scores.view(-1, n_words), sent2[1:].view(-1))
        if back:
            self.stats['xe_costs_bt_%s_%s' % (lang1, lang2)].append(xe_loss.item())
        else:
            self.stats['xe_costs_%s_%s' % (lang1, lang2)].append(xe_loss.item())

        # discriminator feedback loss
        if params.lambda_dis:
            predictions = self.discriminator(encoded.dis_input.view(-1, encoded.dis_input.size(-1)))
            fake_y = torch.LongTensor(predictions.size(0)).random_(1, params.n_langs)
            fake_y = (fake_y + lang1_id) % params.n_langs
            fake_y = fake_y.cuda()
            dis_loss = F.cross_entropy(predictions, fake_y)

        # total loss
        assert lambda_xe > 0
        loss = lambda_xe * xe_loss
        if params.lambda_dis:
            loss = loss + params.lambda_dis * dis_loss

        # check NaN
        if (loss != loss).data.any():
            logger.error("NaN detected")
            exit()

        # optimizer
        self.zero_grad(['enc', 'dec'])
        loss.backward()
        self.update_params(['enc', 'dec'])

        # number of processed sentences / words
        self.stats['processed_s'] += len2.size(0)
        self.stats['processed_w'] += len2.sum()

    def otf_start_multiprocessing(self):
        logger.info("Starting subprocesses for OTF generation ...")

        # initialize subprocesses
        for rank in range(self.num_replicas):
            self.call_async(rank, '_async_otf_init', params=self.params)

    def _async_otf_init(self, rank, device_id, params):
        # build model on subprocess

        from copy import deepcopy
        params = deepcopy(params)
        self.params = params
        self.params.cpu_thread = True
        self.data = None  # do not load data in the CPU threads
        self.iterators = {}
        self.encoder, self.decoder, _, _ = build_mt_model(self.params, self.data, cuda=False)

    def otf_sync_params(self):
        # logger.info("Syncing encoder and decoder params for OTF generation ...")

        def get_flat_params(module):
            return torch._utils._flatten_dense_tensors(
                [p.data for p in module.parameters()])

        encoder_params = get_flat_params(self.encoder).cpu().share_memory_()
        decoder_params = get_flat_params(self.decoder).cpu().share_memory_()

        for rank in range(self.num_replicas):
            self.call_async(rank, '_async_otf_sync_params', encoder_params=encoder_params,
                            decoder_params=decoder_params)

    def _async_otf_sync_params(self, rank, device_id, encoder_params, decoder_params):

        def set_flat_params(module, flat):
            params = [p.data for p in module.parameters()]
            for p, f in zip(params, torch._utils._unflatten_dense_tensors(flat, params)):
                p.copy_(f)

        # copy parameters back into modules
        set_flat_params(self.encoder, encoder_params)
        set_flat_params(self.decoder, decoder_params)

    def otf_bt_gen_async(self, init_cache_size=None):
        logger.info("Populating initial OTF generation cache ...")
        if init_cache_size is None:
            init_cache_size = self.num_replicas
        cache = [
            self.call_async(rank=i % self.num_replicas, action='_async_otf_bt_gen',
                            result_type='otf_gen', fetch_all=True,
                            batches=self.get_worker_batches())
            for i in range(init_cache_size)
        ]
        while True:
            results = cache[0].gen()
            for rank, _ in results:
                cache.pop(0)  # keep the cache a fixed size
                cache.append(
                    self.call_async(rank=rank, action='_async_otf_bt_gen',
                                    result_type='otf_gen', fetch_all=True,
                                    batches=self.get_worker_batches())
                )
            for _, result in results:
                yield result

    def get_worker_batches(self):
        """
        Create batches for CPU threads.
        """
        batches = []

        for direction in self.params.pivo_directions:

            lang1, lang2, lang3 = direction

            # 2-lang back-translation - autoencoding
            if lang1 != lang2 == lang3:
                if self.params.lambda_xe_otfa > 0:
                    (sent1, len1), (sent3, len3) = self.get_batch('otf', lang1, lang3)
            # 2-lang back-translation - parallel data
            elif lang1 == lang3 != lang2:
                if self.params.lambda_xe_otfd > 0:
                    sent1, len1 = self.get_batch('otf', lang1, None)
                    sent3, len3 = sent1, len1
            # 3-lang back-translation - parallel data
            else:
                assert lang1 != lang2 and lang2 != lang3 and lang1 != lang3
                if self.params.lambda_xe_otfd > 0:
                    (sent1, len1), (sent3, len3) = self.get_batch('otf', lang1, lang3)

            batches.append({
                'direction': direction,
                'sent1': sent1,
                'sent3': sent3,
                'len1': len1,
                'len3': len3,
            })

        return batches

    def _async_otf_bt_gen(self, rank, device_id, batches):
        """
        On the fly back-translation (generation step).
        """
        params = self.params
        self.encoder.eval()
        self.decoder.eval()

        results = []

        with torch.no_grad():

            for batch in batches:
                lang1, lang2, lang3 = batch['direction']
                lang1_id = params.lang2id[lang1]
                lang2_id = params.lang2id[lang2]
                sent1, len1 = batch['sent1'], batch['len1']
                sent3, len3 = batch['sent3'], batch['len3']

                # lang1 -> lang2
                encoded = self.encoder(sent1, len1, lang_id=lang1_id)
                max_len = int(1.5 * len1.max() + 10)
                if params.otf_sample == -1:
                    sent2, len2, _ = self.decoder.generate(encoded, lang_id=lang2_id, max_len=max_len)
                else:
                    sent2, len2, _ = self.decoder.generate(encoded, lang_id=lang2_id, max_len=max_len,
                                                           sample=True, temperature=params.otf_sample)

                # keep cached batches on CPU for easier transfer
                assert not any(x.is_cuda for x in [sent1, sent2, sent3])
                results.append(dict([
                    ('lang1', lang1), ('sent1', sent1), ('len1', len1),
                    ('lang2', lang2), ('sent2', sent2), ('len2', len2),
                    ('lang3', lang3), ('sent3', sent3), ('len3', len3),
                ]))

        return (rank, results)

    def otf_bt(self, batch, lambda_xe, backprop_temperature):
        """
        On the fly back-translation.
        """
        params = self.params
        lang1, sent1, len1 = batch['lang1'], batch['sent1'], batch['len1']
        lang2, sent2, len2 = batch['lang2'], batch['sent2'], batch['len2']
        lang3, sent3, len3 = batch['lang3'], batch['sent3'], batch['len3']
        if lambda_xe == 0:
            logger.warning("Unused generated CPU batch for direction %s-%s-%s!" % (lang1, lang2, lang3))
            return
        lang1_id = params.lang2id[lang1]
        lang2_id = params.lang2id[lang2]
        lang3_id = params.lang2id[lang3]
        direction = (lang1, lang2, lang3)
        assert direction in params.pivo_directions
        loss_fn = self.decoder.loss_fn[lang3_id]
        n_words2 = params.n_words[lang2_id]
        n_words3 = params.n_words[lang3_id]
        self.encoder.train()
        self.decoder.train()

        # prepare batch
        sent1, sent2, sent3 = sent1.cuda(), sent2.cuda(), sent3.cuda()
        bs = sent1.size(1)

        if backprop_temperature == -1:
            # lang2 -> lang3
            encoded = self.encoder(sent2, len2, lang_id=lang2_id)
        else:
            # lang1 -> lang2
            encoded = self.encoder(sent1, len1, lang_id=lang1_id)
            scores = self.decoder(encoded, sent2[:-1], lang_id=lang2_id)
            assert scores.size() == (len2.max() - 1, bs, n_words2)

            # lang2 -> lang3
            bos = torch.cuda.FloatTensor(1, bs, n_words2).zero_()
            bos[0, :, params.bos_index[lang2_id]] = 1
            sent2_input = torch.cat([bos, F.softmax(scores / backprop_temperature, -1)], 0)
            encoded = self.encoder(sent2_input, len2, lang_id=lang2_id)

        # cross-entropy scores / loss
        scores = self.decoder(encoded, sent3[:-1], lang_id=lang3_id)
        xe_loss = loss_fn(scores.view(-1, n_words3), sent3[1:].view(-1))
        self.stats['xe_costs_%s_%s_%s' % direction].append(xe_loss.item())
        assert lambda_xe > 0
        loss = lambda_xe * xe_loss

        # check NaN
        if (loss != loss).data.any():
            logger.error("NaN detected")
            exit()

        # optimizer
        assert params.otf_update_enc or params.otf_update_dec
        to_update = []
        if params.otf_update_enc:
            to_update.append('enc')
        if params.otf_update_dec:
            to_update.append('dec')
        self.zero_grad(to_update)
        loss.backward()
        self.update_params(to_update)

        # number of processed sentences / words
        self.stats['processed_s'] += len3.size(0)
        self.stats['processed_w'] += len3.sum()

    def iter(self):
        """
        End of iteration.
        """
        self.n_iter += 1
        self.n_total_iter += 1
        n_batches = len(self.params.mono_directions) + len(self.params.para_directions) + len(self.params.back_directions) + len(self.params.pivo_directions)
        self.n_sentences += n_batches * self.params.batch_size
        self.print_stats()
        update_lambdas(self.params, self.n_total_iter)

    def print_stats(self):
        """
        Print statistics about the training.
        """
        # average loss / statistics
        if self.n_iter % 50 == 0:
            mean_loss = [
                ('DIS', 'dis_costs'),
            ]
            for lang in self.params.mono_directions:
                mean_loss.append(('XE-%s-%s' % (lang, lang), 'xe_costs_%s_%s' % (lang, lang)))
            for lang1, lang2 in self.params.para_directions:
                mean_loss.append(('XE-%s-%s' % (lang1, lang2), 'xe_costs_%s_%s' % (lang1, lang2)))
            for lang1, lang2 in self.params.back_directions:
                mean_loss.append(('XE-BT-%s-%s' % (lang1, lang2), 'xe_costs_bt_%s_%s' % (lang1, lang2)))
            for lang1, lang2, lang3 in self.params.pivo_directions:
                mean_loss.append(('XE-%s-%s-%s' % (lang1, lang2, lang3), 'xe_costs_%s_%s_%s' % (lang1, lang2, lang3)))
            for lang in self.params.langs:
                mean_loss.append(('LME-%s' % lang, 'lme_costs_%s' % lang))
                mean_loss.append(('LMD-%s' % lang, 'lmd_costs_%s' % lang))
                mean_loss.append(('LMER-%s' % lang, 'lmer_costs_%s' % lang))
                mean_loss.append(('ENC-L2-%s' % lang, 'enc_norms_%s' % lang))

            s_iter = "%7i - " % self.n_iter
            s_stat = ' || '.join(['{}: {:7.4f}'.format(k, np.mean(self.stats[l]))
                                 for k, l in mean_loss if len(self.stats[l]) > 0])
            for _, l in mean_loss:
                del self.stats[l][:]

            # processing speed
            new_time = time.time()
            diff = new_time - self.last_time
            s_speed = "{:7.2f} sent/s - {:8.2f} words/s - ".format(self.stats['processed_s'] * 1.0 / diff,
                                                                   self.stats['processed_w'] * 1.0 / diff)
            self.stats['processed_s'] = 0
            self.stats['processed_w'] = 0
            self.last_time = new_time

            lrs = self.get_lrs(['enc', 'dec'])
            s_lr = " - LR " + ",".join("{}={:.4e}".format(k, lr) for k, lr in lrs.items())

            # generation time
            if len(self.params.pivo_directions) > 0:
                s_time = " - Sentences generation time: % .2fs (%.2f%%)" % (self.gen_time, 100. * self.gen_time / diff)
                self.gen_time = 0
            else:
                s_time = ""

            # log speed + stats
            logger.info(s_iter + s_speed + s_stat + s_lr + s_time)

    def save_model(self, name):
        """
        Save the model.
        """
        path = os.path.join(self.params.dump_path, '%s.pth' % name)
        logger.info('Saving model to %s ...' % path)
        torch.save({
            'enc': self.encoder,
            'dec': self.decoder,
            'dis': self.discriminator,
            'lm': self.lm,
        }, path)

    def save_checkpoint(self):
        """
        Checkpoint the experiment.
        """
        checkpoint_data = {
            'encoder': self.encoder,
            'decoder': self.decoder,
            'discriminator': self.discriminator,
            'lm': self.lm,
            'enc_optimizer': self.enc_optimizer,
            'dec_optimizer': self.dec_optimizer,
            'dis_optimizer': self.dis_optimizer,
            'lm_optimizer': self.lm_optimizer,
            'epoch': self.epoch,
            'n_total_iter': self.n_total_iter,
            'best_metrics': self.best_metrics,
            'best_stopping_criterion': self.best_stopping_criterion,
        }
        checkpoint_path = os.path.join(self.params.dump_path, 'checkpoint.pth')
        logger.info("Saving checkpoint to %s ..." % checkpoint_path)
        torch.save(checkpoint_data, checkpoint_path)

    def reload_checkpoint(self):
        """
        Reload a checkpoint if we find one.
        """
        # reload checkpoint
        checkpoint_path = os.path.join(self.params.dump_path, 'checkpoint.pth')
        if not os.path.isfile(checkpoint_path):
            return
        logger.warning('Reloading checkpoint from %s ...' % checkpoint_path)
        checkpoint_data = torch.load(checkpoint_path)
        self.encoder = checkpoint_data['encoder']
        self.decoder = checkpoint_data['decoder']
        self.discriminator = checkpoint_data['discriminator']
        self.lm = checkpoint_data['lm']
        self.enc_optimizer = checkpoint_data['enc_optimizer']
        self.dec_optimizer = checkpoint_data['dec_optimizer']
        self.dis_optimizer = checkpoint_data['dis_optimizer']
        self.lm_optimizer = checkpoint_data['lm_optimizer']
        self.epoch = checkpoint_data['epoch']
        self.n_total_iter = checkpoint_data['n_total_iter']
        self.best_metrics = checkpoint_data['best_metrics']
        self.best_stopping_criterion = checkpoint_data['best_stopping_criterion']
        self.model_opt = {
            'enc': (self.encoder, self.enc_optimizer),
            'dec': (self.decoder, self.dec_optimizer),
            'dis': (self.discriminator, self.dis_optimizer),
            'lm': (self.lm, self.lm_optimizer),
        }
        logger.warning('Checkpoint reloaded. Resuming at epoch %i ...' % self.epoch)

    def test_sharing(self):
        """
        Test to check that parameters are shared correctly.
        """
        test_sharing(self.encoder, self.decoder, self.lm, self.params)
        logger.info("Test: Parameters are shared correctly.")

    def save_best_model(self, scores):
        """
        Save best models according to given validation metrics.
        """
        for metric in self.VALIDATION_METRICS:
            if scores[metric] > self.best_metrics[metric]:
                self.best_metrics[metric] = scores[metric]
                logger.info('New best score for %s: %.6f' % (metric, scores[metric]))
                self.save_model('best-%s' % metric)

    def save_periodic(self):
        """
        Save the models periodically.
        """
        if self.params.save_periodic and self.epoch % 20 == 0 and self.epoch > 0:
            self.save_model('periodic-%i' % self.epoch)

    def end_epoch(self, scores):
        """
        End the epoch.
        """
        # stop if the stopping criterion has not improved after a certain number of epochs
        if self.stopping_criterion is not None:
            assert self.stopping_criterion in scores
            if scores[self.stopping_criterion] > self.best_stopping_criterion:
                self.best_stopping_criterion = scores[self.stopping_criterion]
                logger.info("New best validation score: %f" % self.best_stopping_criterion)
                self.decrease_counts = 0
            if scores[self.stopping_criterion] < self.best_stopping_criterion:
                logger.info("Not a better validation score (%i / %i)."
                            % (self.decrease_counts, self.decrease_counts_max))
                self.decrease_counts += 1
            if self.decrease_counts > self.decrease_counts_max:
                logger.info("Stopping criterion has been below its best value more "
                            "than %i epochs. Ending the experiment..." % self.decrease_counts_max)
                exit()
        self.epoch += 1
        self.save_checkpoint()
