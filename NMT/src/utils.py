# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import re
import sys
import pickle
import random
import inspect
import argparse
import subprocess
from logging import getLogger
import numpy as np
import torch
from torch import optim

from .logger import create_logger
from .data.dictionary import EOS_WORD, UNK_WORD
from .adam_inverse_sqrt_with_warmup import AdamInverseSqrtWithWarmup


logger = getLogger()


FALSY_STRINGS = {'off', 'false', '0'}
TRUTHY_STRINGS = {'on', 'true', '1'}


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")


def initialize_exp(params, logger_filename='train.log'):
    """
    Initialize the experience:
    - dump parameters
    - create a logger
    - set the random seed
    """
    # dump parameters
    get_dump_path(params)
    pickle.dump(params, open(os.path.join(params.dump_path, 'params.pkl'), 'wb'))

    # get running command
    command = ["python", sys.argv[0]]
    for x in sys.argv[1:]:
        if x.startswith('--'):
            assert '"' not in x and "'" not in x
            command.append(x)
        else:
            assert "'" not in x
            command.append("'%s'" % x)
    command = ' '.join(command)
    params.command = command + ' --exp_id "%s"' % params.exp_id

    # random seed
    if params.seed >= 0:
        np.random.seed(params.seed)
        torch.manual_seed(params.seed)
        torch.cuda.manual_seed(params.seed)

    # environment variables
    if 'pivo_directions' in params and len(params.pivo_directions) > 0:
        os.environ["OMP_NUM_THREADS"] = "2"
        os.environ["MKL_NUM_THREADS"] = "2"

    # create a logger
    logger = create_logger(os.path.join(params.dump_path, logger_filename))
    logger.info('============ Initialized logger ============')
    logger.info('\n'.join('%s: %s' % (k, str(v))
                          for k, v in sorted(dict(vars(params)).items())))
    logger.info('The experiment will be stored in %s\n' % params.dump_path)
    logger.info('Running command: %s\n' % params.command)
    return logger


def get_dump_path(params):
    """
    Create a directory to store the experiment.
    """
    assert len(params.exp_name) > 0
    dump_path = './' if params.dump_path == '' else params.dump_path
    subprocess.Popen("mkdir -p %s" % dump_path, shell=True).wait()
    assert os.path.isdir(dump_path)

    # create the sweep path if it does not exist
    sweep_path = os.path.join(dump_path, params.exp_name)
    if not os.path.exists(sweep_path):
        subprocess.Popen("mkdir %s" % sweep_path, shell=True).wait()

    # create an ID for the job if it is not given in the parameters.
    # if we run on the cluster, the job ID is the one of Chronos.
    # otherwise, it is randomly generated
    if params.exp_id == '':
        exp_id = os.environ.get('CHRONOS_JOB_ID')
        if exp_id is None:
            exp_id = os.environ.get('SLURM_JOB_ID')
        if exp_id is None:
            chars = 'abcdefghijklmnopqrstuvwxyz0123456789'
            while True:
                exp_id = ''.join(random.choice(chars) for _ in range(10))
                if not os.path.isdir(os.path.join(sweep_path, exp_id)):
                    break
        else:
            assert exp_id.isdigit()
        params.exp_id = exp_id
    else:
        assert os.path.isdir(os.path.join(sweep_path, params.exp_id))  # reload an experiment

    # create the dump folder / update parameters
    params.dump_path = os.path.join(sweep_path, params.exp_id)
    if not os.path.isdir(params.dump_path):
        subprocess.Popen("mkdir %s" % params.dump_path, shell=True).wait()


def get_optimizer(parameters, s):
    """
    Parse optimizer parameters.
    Input should be of the form:
        - "sgd,lr=0.01"
        - "adagrad,lr=0.1,lr_decay=0.05"
    """
    if "," in s:
        method = s[:s.find(',')]
        optim_params = {}
        for x in s[s.find(',') + 1:].split(','):
            split = x.split('=')
            assert len(split) == 2
            assert re.match("^[+-]?(\d+(\.\d*)?|\.\d+)$", split[1]) is not None
            optim_params[split[0]] = float(split[1])
    else:
        method = s
        optim_params = {}

    if method == 'adadelta':
        optim_fn = optim.Adadelta
    elif method == 'adagrad':
        optim_fn = optim.Adagrad
    elif method == 'adam':
        optim_fn = optim.Adam
        optim_params['betas'] = (optim_params.get('beta1', 0.5), optim_params.get('beta2', 0.999))
        optim_params.pop('beta1', None)
        optim_params.pop('beta2', None)
    elif method == 'adamax':
        optim_fn = optim.Adamax
    elif method == 'asgd':
        optim_fn = optim.ASGD
    elif method == 'rmsprop':
        optim_fn = optim.RMSprop
    elif method == 'rprop':
        optim_fn = optim.Rprop
    elif method == 'sgd':
        optim_fn = optim.SGD
        assert 'lr' in optim_params
    elif method == 'adam_inverse_sqrt':
        optim_fn = AdamInverseSqrtWithWarmup
        optim_params['betas'] = (optim_params.get('beta1', 0.9), optim_params.get('beta2', 0.98))
        optim_params['warmup_updates'] = optim_params.get('warmup_updates', 4000)
        optim_params.pop('beta1', None)
        optim_params.pop('beta2', None)
    else:
        raise Exception('Unknown optimization method: "%s"' % method)

    # check that we give good parameters to the optimizer
    expected_args = inspect.getargspec(optim_fn.__init__)[0]
    assert expected_args[:2] == ['self', 'params']
    if not all(k in expected_args[2:] for k in optim_params.keys()):
        raise Exception('Unexpected parameters: expected "%s", got "%s"' % (
            str(expected_args[2:]), str(optim_params.keys())))

    return optim_fn(parameters, **optim_params)


def reload_parameters(old_params, new_params, attributes):
    """
    Reload the parameters of a previous model.
    """
    for k, v in old_params.__dict__.items():
        if k in attributes and k not in new_params:
            setattr(new_params, k, v)


def reload_model(model, to_reload, attributes):
    """
    Reload a previously trained model.
    """
    # check parameters sizes
    model_params = set(model.state_dict().keys())
    to_reload_params = set(to_reload.state_dict().keys())
    assert model_params == to_reload_params, (model_params - to_reload_params,
                                              to_reload_params - model_params)

    # check attributes
    warnings = []
    errors = []
    for k in attributes:
        assert type(k) is tuple or type(k) is str
        k, strict = k if type(k) is tuple else (k, True)
        if getattr(model, k, None) is None:
            errors.append('- Attribute "%s" not found in the current model' % k)
        if getattr(to_reload, k, None) is None:
            errors.append('- Attribute "%s" not found in the model to reload' % k)
        if getattr(model, k, None) != getattr(to_reload, k, None):
            message = ('- Attribute "%s" differs between the current model (%s) '
                       'and the one to reload (%s)'
                       % (k, str(getattr(model, k)), str(getattr(to_reload, k))))
            (errors if strict else warnings).append(message)
    if len(warnings) > 0:
        logger.warning('Different parameters:\n%s' % '\n'.join(warnings))
    if len(errors) > 0:
        logger.error('Incompatible parameters:\n%s' % '\n'.join(errors))
        exit()

    # copy saved parameters
    for k in model.state_dict().keys():
        if model.state_dict()[k].size() != to_reload.state_dict()[k].size():
            raise Exception("Expected tensor {} of size {}, but got {}".format(
                k, model.state_dict()[k].size(),
                to_reload.state_dict()[k].size()
            ))
        model.state_dict()[k].copy_(to_reload.state_dict()[k])


def clip_parameters(model, clip):
    """
    Clip model weights.
    """
    if clip > 0:
        for x in model.parameters():
            x.data.clamp_(-clip, clip)


def get_grad_norm(model):
    """
    Return the norm of the parameters gradients.
    """
    norm = 0
    for param in model.parameters():
        norm += param.grad.data.norm(2) ** 2
    return np.sqrt(norm)


def parse_lambda_config(params, name):
    """
    Parse the configuration of lambda coefficient (for scheduling).
    x = "3"                  # lambda will be a constant equal to x
    x = "0:1,1000:0"         # lambda will start from 1 and linearly decrease to 0 during the first 1000 iterations
    x = "0:0,1000:0,2000:1"  # lambda will be equal to 0 for the first 1000 iterations, then will linearly increase to 1 until iteration 2000
    """
    x = getattr(params, name)
    split = x.split(',')
    if len(split) == 1:
        setattr(params, name, float(x))
        setattr(params, name + '_config', None)
    else:
        split = [s.split(':') for s in split]
        assert all(len(s) == 2 for s in split)
        assert all(k.isdigit() for k, _ in split)
        assert all(int(split[i][0]) < int(split[i + 1][0]) for i in range(len(split) - 1))
        setattr(params, name, float(split[0][1]))
        setattr(params, name + '_config', [(int(k), float(v)) for k, v in split])


def update_lambda_value(config, n_iter):
    """
    Update a lambda value according to its schedule configuration.
    """
    ranges = [i for i in range(len(config) - 1) if config[i][0] <= n_iter < config[i + 1][0]]
    if len(ranges) == 0:
        assert n_iter >= config[-1][0]
        return config[-1][1]
    assert len(ranges) == 1
    i = ranges[0]
    x_a, y_a = config[i]
    x_b, y_b = config[i + 1]
    return y_a + (n_iter - x_a) * float(y_b - y_a) / float(x_b - x_a)


def update_lambdas(params, n_total_iter):
    """
    Update all lambda coefficients.
    """
    if params.lambda_xe_mono_config is not None:
        params.lambda_xe_mono = update_lambda_value(params.lambda_xe_mono_config, n_total_iter)
    if params.lambda_xe_para_config is not None:
        params.lambda_xe_para = update_lambda_value(params.lambda_xe_para_config, n_total_iter)
    if params.lambda_xe_back_config is not None:
        params.lambda_xe_back = update_lambda_value(params.lambda_xe_back_config, n_total_iter)
    if params.lambda_xe_otfd_config is not None:
        params.lambda_xe_otfd = update_lambda_value(params.lambda_xe_otfd_config, n_total_iter)
    if params.lambda_xe_otfa_config is not None:
        params.lambda_xe_otfa = update_lambda_value(params.lambda_xe_otfa_config, n_total_iter)
    if params.lambda_dis_config is not None:
        params.lambda_dis = update_lambda_value(params.lambda_dis_config, n_total_iter)
    if params.lambda_lm_config is not None:
        params.lambda_lm = update_lambda_value(params.lambda_lm_config, n_total_iter)


def get_mask(lengths, all_words, expand=None, ignore_first=False, batch_first=False, cuda=True):
    """
    Create a mask of shape (slen, bs) or (bs, slen).
    """
    bs, slen = lengths.size(0), lengths.max()
    mask = torch.ByteTensor(slen, bs).zero_()
    for i in range(bs):
        if all_words:
            mask[:lengths[i], i] = 1
        else:
            mask[lengths[i] - 1, i] = 1
    if expand is not None:
        assert type(expand) is int
        mask = mask.unsqueeze(2).expand(slen, bs, expand)
    if ignore_first:
        mask[0].fill_(0)
    if batch_first:
        mask = mask.transpose(0, 1)
    if cuda:
        mask = mask.cuda()
    return mask


def reverse_sentences(batch, lengths):
    """
    Reverse sentences inside a batch.
    """
    bs = lengths.size(0)
    assert batch.size(1) == bs
    new_batch = batch.clone()
    inv_idx = torch.arange(lengths.max() - 1, -1, -1)
    for i in range(bs):
        new_batch[:lengths[i], i].copy_(new_batch[:, i][inv_idx[-lengths[i]:]])
    return new_batch


def restore_segmentation(path):
    """
    Take a file segmented with BPE and restore it to its original segmentation.
    """
    assert os.path.isfile(path)
    restore_cmd = "sed -i -r 's/(@@ )|(@@ ?$)//g' %s"
    subprocess.Popen(restore_cmd % path, shell=True).wait()


def create_word_masks(params, data):
    """
    Create masks for allowed / forbidden output words.
    """
    if not hasattr(params, 'vocab') or len(params.vocab) == 0:
        return
    params.vocab_mask_pos = []
    params.vocab_mask_neg = []
    for lang, n_words in zip(params.langs, params.n_words):
        dico = data['dico'][lang]
        vocab = data['vocab'][lang]
        words = [EOS_WORD, UNK_WORD] + list(vocab)
        mask_pos = set([dico.index(w) for w in words])
        mask_neg = [i for i in range(n_words) if i not in mask_pos]
        params.vocab_mask_pos.append(torch.LongTensor(sorted(mask_pos)))
        params.vocab_mask_neg.append(torch.LongTensor(sorted(mask_neg)))
