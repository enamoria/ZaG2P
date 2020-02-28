# -*- coding: utf-8 -*-

""" Created on 10:30 AM, 9/4/19
    @author: ngunhuconchocon
    @brief: Пролетарии всех стран, соединяйтесь! да здравствует наша советская родина
"""
from __future__ import print_function, division, absolute_import

import argparse
import os
import time

import dill as pickle
import torch
import torchtext.data as data

from codecs import open

from .DictClass import VNDict
from .constant import parser, project_root
from .models import G2P
from .utils import read_dict, uncombine_phonemes_tone_with_space

UNVOICE_SOUND = "6"
nucleus = ["aa", "ee", "ea", "oa", "aw", "ie", "uo", "a", "wa", "oo", "e", "i", "o", "u", "ow", "uw", "w", ""]
coda = ["iz", "pc", "nz", "tc", "ngz", "kc", "uz", "mz"]

# Unvoiced sound
unvoiced_sound = ["b", "c", "ch", "đ", "g", "k", "p", "x", "s", "t", "f"]


# def read_dict(dictpath):
#     """
#         this dict is different: dict[phoneme] = word, like dict['5 d i t'] = địt
#     :param dictpath:
#     :return:
#     """
#     vietdict = {'6 b': '(bờ)', '6 k': '(cờ)', '6 tr': '(chờ)', '6 d': '(dờ)', '6 dd': '(đờ)', '6 g': '(gờ)', '6 l': '(lờ)', '6 m': '(mờ)', '6 n': '(nờ)', '6 p': '(pờ)', '6 ph': '(phờ)', '6 r': '(rờ)', '6 s': '(xờ)', '6 t': '(tờ)', '6 th': '(thờ)', '6 v': '(vờ'}
#     with open(dictpath, "r", encoding="utf8") as f:
#         for line in f.readlines():
#             if line and line.strip() and line[0] != "#":
#                 temp = line.strip().split(" ")
#                 word, phonemes = temp[0], temp[1:]
#
#                 vietdict[" ".join(phonemes)] = word
#     return vietdict


def predict(batch, model):
    p_field = batch.dataset.fields['phoneme']
    prediction = model(batch.grapheme).tolist()[:-1]
    phonemes = ' '.join([p_field.vocab.itos[p] for p in prediction])
    # uncombined_phonemes = uncombine_phonemes_tone(phonemes, None)
    uncombined_phonemes = uncombine_phonemes_tone_with_space(phonemes, None)
    return uncombined_phonemes


def convert_from_phonemes_to_syllables(batch, model, vietdict):
    uncombined_phonemes = predict(batch, model)

    prev = 0
    syllables = []

    for i, phoneme_or_tone in enumerate(uncombined_phonemes):
        if phoneme_or_tone.isdigit() and i > 1:
            syllables.append(vietdict[" ".join(uncombined_phonemes[prev:i])])
            prev = i
        elif i == len(uncombined_phonemes) - 1:
            syllables.append(vietdict[" ".join(uncombined_phonemes[prev:])])

    return " ".join(syllables)


def combine_phoneme_and_tone(phonemes_and_tone):
    # phonemes, tone = phonemes_and_tone[1:], phonemes_and_tone[0]
    prev_pos = 0
    current_tone = phonemes_and_tone[0]
    phonemes = []
    for i, item in enumerate(phonemes_and_tone[1:]):
        if item.isdigit():
            prev_pos = i
            current_tone = item
        else:
            if item in nucleus or current_tone == UNVOICE_SOUND:
                phonemes.append(item + current_tone)
            else:
                phonemes.append(item)

    return phonemes


def load_model(fields_path=None, model_path=None, dict_path=None):
    args = argparse.Namespace(**parser)
    config = args

    # g_field = data.Field(init_token='<s>', tokenize=(lambda x: list(x.split()[0])[::-1]))
    # p_field = data.Field(init_token='<os>', eos_token='</os>',
    #                      tokenize=(lambda x: x.split()))
    #
    # filepath = os.path.join(project_root, os.path.join("tts_dict_prepare", 'oov_syllable_new_type_1'))
    # train_data, val_data, test_data, combined_data, all_data, _ = VNDict.splits(filepath, g_field, p_field, 1234)
    #
    # g_field.build_vocab(train_data, val_data, test_data)
    # p_field.build_vocab(train_data, val_data, test_data)

    if not fields_path:
        # We have to manually redeclare fields, or the code will only runnable in python 3.7
        fields_path = os.path.join(project_root, os.path.join(config.intermediate_path, "gp_fields.pkl"))
        
    if not model_path:
        model_path = os.path.join(project_root, os.path.join(config.intermediate_path, "best_model_adagrad_attn.pth"))
    if not dict_path:
        dict_path = os.path.join(project_root, "tts_dict_prepare/vn.dict")

    with open(fields_path, "rb") as f:
        fields = pickle.load(f)
        g_field, p_field, config = fields['g_field'], fields['p_field'], fields['config']
        # _, _, config = fields['g_field'], fields['p_field'], fields['config']

    model = G2P(config)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    if config.cuda:
        model.cuda()

    vietdict = read_dict(dictpath=dict_path)
    return (g_field, p_field, model), vietdict


def G2S(word, model_and_fields, vietdict, use_cuda=True, return_phoneme=False, combine_tone_phone=True):
    """
        Convert grapheme to syllables
    :param combine_tone_phone: Whether to combine tone with phoneme. Only be taken in to account if return_phoneme = True
    :param return_phoneme: if true, return list of phoneme
    :param word: string
    :param model_and_fields: model, getting from load_model(). Note that this contain g_field, p_field, and g2p model
    :param vietdict: vn.dict :D
    :param use_cuda:
    :return:
    """
    try:
        device = -1
        if use_cuda:
            if torch.cuda.is_available():
                device = "cuda"
        word = word + " xxxxx"
        g_field, p_field, model = model_and_fields
        test_data = VNDict([word], g_field, p_field)
        test_iter = data.Iterator(test_data, batch_size=1,
                                  train=False, shuffle=True, device=device)

        results = []

        if return_phoneme:
            for batch in test_iter:
                grapheme = batch.grapheme.squeeze(1).data.tolist()[1:][::-1]
                grapheme = ''.join([g_field.vocab.itos[g] for g in grapheme])

                if combine_tone_phone:
                    results.append("{} {}".format(grapheme, " ".join(combine_phoneme_and_tone(predict(batch, model)))))
                else:
                    results.append("{} {}".format(grapheme, " ".join(predict(batch, model))))
        else:
            for batch in test_iter:
                grapheme = batch.grapheme.squeeze(1).data.tolist()[1:][::-1]
                grapheme = ''.join([g_field.vocab.itos[g] for g in grapheme])
                results.append("{} {}".format(grapheme, convert_from_phonemes_to_syllables(batch, model, vietdict)))

        return results
    except Exception as e:
        return word.split(" ")[0]


if __name__ == "__main__":
    model, vietdict = load_model()

    start = time.time()
    # print(G2S("software", model, vietdict))
    # print(G2S("developing", model, vietdict))
    # print(G2S("seasoning", model, vietdict))
    # print(G2S("versioning", model, vietdict))
    # print(G2S("zika", model, vietdict))
    # print(G2S("zalo", model, vietdict, return_phoneme=False))

    for _ in range(100):
        print(G2S("leganes", model, vietdict))

    print("Elapsed time: {}".format(time.time() - start))
