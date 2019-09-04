# -*- coding: utf-8 -*-

""" Created on 10:30 AM, 9/4/19
    @author: ngunhuconchocon
    @brief: Пролетарии всех стран, соединяйтесь! да здравствует наша советская родина
"""

from DictClass import CMUDict, VNDict
from models import G2P
import argparse
import dill as pickle

import pdb
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext.data as data
import Levenshtein  # https://github.com/ztane/python-Levenshtein/
from utils import uncombine_phonemes_tone, map_phone_to_unvoice
from constant import parser, project_root

parser['intermediate_path'] = 'intermediate/g2p_vi/'  # path to save models
parser['beam_size'] = 30  # size of beam for beam-search
parser['d_embed'] = 1000  # embedding dimension
parser['d_hidden'] = 1000  # hidden dimension
parser['epochs'] = 15
parser['max_len'] = 30  # max length of grapheme/phoneme sequences
parser['lr'] = 0.01
parser['lr_min'] = 1e-5

args = argparse.Namespace(**parser)
config = args
tone_of_unvoiced_phoneme = "6"


def read_dict(dictpath):
    """
        this dict is different: dict[phoneme] = word, like dict['5 d i t'] = địt
    :param dictpath:
    :return:
    """
    vietdict = {'6 b': '(bờ)', '6 k': '(cờ)', '6 tr': '(chờ)', '6 d': '(dờ)', '6 dd': '(đờ)', '6 g': '(gờ)', '6 l': '(lờ)', '6 m': '(mờ)', '6 n': '(nờ)', '6 p': '(pờ)', '6 ph': '(phờ)', '6 r': '(rờ)', '6 s': '(xờ)', '6 t': '(tờ)', '6 th': '(thờ)', '6 v': '(vờ'}
    with open(dictpath, "r") as f:
        for line in f.readlines():
            if line and line.strip() and line[0] != "#":
                temp = line.strip().split(" ")
                word, phonemes = temp[0], temp[1:]

                vietdict[" ".join(phonemes)] = word
    return vietdict


def convert_from_phonemes_to_syllables(batch, model, vietdict):
    p_field = batch.dataset.fields['phoneme']
    prediction = model(batch.grapheme).data.tolist()[:-1]

    phonemes = ' '.join([p_field.vocab.itos[p] for p in prediction])
    uncombined_phonemes = uncombine_phonemes_tone(phonemes, None)

    prev = 0
    syllables = []

    for i, phoneme_or_tone in enumerate(uncombined_phonemes):
        if phoneme_or_tone.isdigit() and i > 1:
            syllables.append(vietdict[" ".join(uncombined_phonemes[prev:i])])
            prev = i
        elif i == len(uncombined_phonemes) - 1:
            syllables.append(vietdict[" ".join(uncombined_phonemes[prev:])])

    return " ".join(syllables)


def load_model(fields_path, model_path, dict_path):
    with open(fields_path, "rb") as f:
        fields = pickle.load(f)
        g_field, p_field, config = fields['g_field'], fields['p_field'], fields['config']

    model = G2P(config)
    model.load_state_dict(torch.load(model_path))

    if config.cuda:
        model.cuda()

    vietdict = read_dict(dictpath=dict_path)
    return (g_field, p_field, model), vietdict


def G2S(word, model_and_fields, vietdict, use_cuda=True):
    """
        Convert grapheme to syllables
    :param word: string
    :param model_and_fields: model, getting from load_model(). Note that this contain g_field, p_field, and g2p model
    :param vietdict: vn.dict :D
    :param use_cuda:
    :return:
    """
    device = -1
    if use_cuda:
        if torch.cuda.is_available():
            device = "cuda"
    word = word + " x x x"
    g_field, p_field, model = model_and_fields
    test_data = VNDict([word], g_field, p_field)
    test_iter = data.Iterator(test_data, batch_size=1,
                              train=False, shuffle=True, device=device)

    results = []
    for batch in test_iter:
        grapheme = batch.grapheme.squeeze(1).data.tolist()[1:][::-1]
        grapheme = ''.join([g_field.vocab.itos[g] for g in grapheme])
        results.append("{} {}".format(grapheme, convert_from_phonemes_to_syllables(batch, model, vietdict)))
        print(results[-1])

    return results


if __name__ == "__main__":
    fields_path = os.path.join(project_root, os.path.join(config.intermediate_path, "gp_fields.pkl"))
    model_path = os.path.join(project_root, os.path.join(config.intermediate_path, "best_model_adagrad_attn.pth"))
    dict_path = os.path.join(project_root, "tts_dict_prepare/vn.dict")

    model, vietdict = load_model(fields_path=fields_path, model_path=model_path, dict_path=dict_path)

    start = time.time()
    G2S("fuck", model, vietdict)
    print("Elapsed time: {}".format(time.time() - start))
