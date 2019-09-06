# -*- coding: utf-8 -*-

""" Created on 9:25 AM, 8/19/19
    @author: ngunhuconchocon
    @brief: Пролетарии всех стран, соединяйтесь! да здравствует наша советская родина
"""

from visdom import Visdom
import numpy as np
import re

coda_nucleus_and_semivowel = ['iz', 'pc', 'nz', 'tc', 'ngz', 'kc', 'uz', 'mz', 'aa', 'ee', 'ea', 'oa', 'aw', 'ie', 'uo', 'a', 'wa', 'oo', 'e', 'i',
                              'o', 'u', 'ow', 'uw', 'w']

coda = ["iz", "pc", "nz", "tc", "ngz", "kc", "uz", "mz"]
nucleus = ["aa", "ee", "ea", "oa", "aw", "ie", "uo", "a", "wa", "oo", "e", "i", "o", "u", "ow", "uw"]
initial = ["nh", "tr", "ng", "th", "ph", "dd", "d", "b", "g", "h", "k", "kh", "m", "l", "n", "p", "s", "r", "t", "v"]

tone_for_unvoiced_sound = "6"  # 6 vietnamese tone is 0-5, this is the 6th "tone"
map_phone_to_unvoice = {'6 b': 'bờ', '6 k': 'cờ', '6 tr': 'chờ', '6 d': 'dờ', '6 dd': 'đờ', '6 g': 'gờ', '6 l': 'lờ', '6 m': 'mờ', '6 n': 'nờ',
                        '6 p': 'pờ', '6 ph': 'phờ', '6 r': 'rờ', '6 s': 'xờ', '6 t': 'tờ', '6 th': 'thờ', '6 v': 'vờ'}


# This is for 8tone dictionary
# coda_nucleus_and_semivowel = ['ngmz', 'ie', 'uo', 'nhz', 'tz', 'pau', 'pz', 'nz', 'aX', 'WX', 'ngz', 'E', 'jz', 'OX', 'O', 'EX', 'W', 'wz', 'kcz', 'a', 'e', 'i', 'o', 'MW', 'u', 'w', 'kz', 'z', 'kpz', 'mz']


class VisdomLinePlotter(object):
    """Plots to Visdom"""

    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}

    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x, x]), Y=np.array([y, y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update='append')


def uncombine_phonemes_tone(phoneme_list_string, grapheme):
    """
        This function seperate the phone_tone combination
        For example:    l ie0 nz0 -> 0 l ie nz (liên)
    :param grapheme: the word that need tobe transcribed
    :param phoneme_list_string: a string: list of phonemes, seperated by <space>
    :return:
    """
    # This part is to deal with unpronounable word, <pad> and </os>. More data solves, but we haven't had it yet
    # TODO REMOVE IN FUTURE
    phoneme_list_string = re.sub(r" +", " ", phoneme_list_string.replace("</os>", "").replace("<pad>", "")).strip()
    # END OF TODO

    result_phonemes = []
    phoneme_list = phoneme_list_string.split()
    current_word = ""
    pos = 0
    for i, phoneme in enumerate(phoneme_list):
        if tone_for_unvoiced_sound in phoneme:
            result_phonemes.extend([tone_for_unvoiced_sound, phoneme[:-1]])
            pos = i + 1
        else:
            if phoneme == "i0":
                pass
            # if :
            #     current_word += phoneme_list[i][-1] + " " + " ".join([temp[:-1] if any(char.isdigit() for char in temp) else temp for temp in phoneme_list[pos:i+1]])
            if (i < len(phoneme_list) - 1 and phoneme[:-1] in nucleus and phoneme_list[i + 1][:-1] in nucleus) or \
                    (i == len(phoneme_list) - 1 or
                     (phoneme[:-1] in coda or phoneme[:-1] in nucleus and
                      (phoneme_list[i + 1] in initial or tone_for_unvoiced_sound in phoneme_list[i + 1] or
                       (phoneme_list[i + 1][-1].isdigit() and phoneme_list[i + 1][-1] != phoneme[-1])))):
                current_word += phoneme_list[i][-1] + " " + " ".join(
                    [temp[:-1] if any(char.isdigit() for char in temp) else temp for temp in phoneme_list[pos:i + 1]])
                result_phonemes.extend(current_word.strip().split())
                current_word = ""
                pos = i + 1

    return result_phonemes
    # return " ".join(result_phonemes)
