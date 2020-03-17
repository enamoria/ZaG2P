# -*- coding: utf-8 -*-

""" Created on 9:25 AM, 8/19/19
    @author: ngunhuconchocon
    @brief: Пролетарии всех стран, соединяйтесь! да здравствует наша советская родина
"""

from visdom import Visdom
import numpy as np
import re
from .constant import coda, nucleus, initial, coda_nucleus_and_semivowel, tone_for_unvoiced_sound


unvoiced_dict = {'b': 'b', 'k': 'k', 'd': 'd', 'đ': 'dd', 'g': 'g', 'p': 'p', 's': 's', 'x': 's', 't': 't', 'v': 'v'}
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


def uncombine_phonemes_tone_with_space(phoneme_list_string, grapheme):
    """
        This function seperate the phone_tone combination, with space character in the predicted phonemes
        For example:    l ie0 nz0 space k ee4 tc -> 0 l ie nz 4 k ee tc (which will be later transcribed into "liên kết")
    :param grapheme: the word that need tobe transcribed
    :param phoneme_list_string: a string: list of phonemes, seperated by <space>
    :return:
    """
    # This part is to deal with unpronounable word, <pad> and </os>. More data solves, but we haven't had it yet
    # TODO REMOVE IN FUTURE
    phoneme_list_string = re.sub(r" +", " ", phoneme_list_string.replace("</os>", "").replace("<pad>", "")).strip()
    # END OF TODO

    result_phonemes = []
    phoneme_list = [temp.strip().split(" ") for temp in phoneme_list_string.split("space")]
    for word in phoneme_list:
        temp_transcribe = []
        for i, phoneme in enumerate(word):
            if phoneme[-1].isdigit():
                temp_transcribe.append(phoneme[:-1])
                result_phonemes.append(phoneme[-1])
            else:
                temp_transcribe.append(phoneme)

        result_phonemes.extend(temp_transcribe)

    return result_phonemes


def read_dict(dictpath):
    """
        this dict is different: dict[phoneme] = word, like dict['5 d i t'] = địt
    :param dictpath:
    :return:
    """
    vietdict = {'6 b': '(bờ)', '6 k': '(cờ)', '6 tr': '(chờ)', '6 d': '(dờ)', '6 dd': '(đờ)', '6 g': '(gờ)', '6 l': '(lờ)', '6 m': '(mờ)',
                '6 n': '(nờ)', '6 p': '(pờ)', '6 ph': '(phờ)', '6 r': '(rờ)', '6 s': '(xờ)', '6 t': '(tờ)', '6 th': '(thờ)', '6 v': '(vờ)'}
    with open(dictpath, "r") as f:
        for line in f.readlines():
            if line and line.strip() and line[0] != "#":
                # Có nghĩa là có trong từ điển và được chấp nhận là 1 từ tiếng Việt, không phải chỉ đơn giản là ghép phụ âm đầu và vần.
                # Ví dụ: đỵt phát âm giống địt nhưng sẽ không được coi là một từ tiếng Việt
                is_real_vietnamese = True

                if "notavailable" in line:
                    is_real_vietnamese = False  # In new dict, words which are not originally in Vietnamese dictionary are noted as "notavailable"

                line = line.strip().replace("notavailable", "").replace("completelynotavailable", "")
                temp = line.strip().split(" ")
                word, phonemes = temp[0], temp[1:]

                if word == "ning" or word == "véc":
                    print(1)
                # if "gia" in line:
                #     print(1)

                phonemes_string = " ".join(phonemes)
                if phonemes_string not in vietdict:
                    vietdict[phonemes_string] = word

                if (("ing" not in word or word[0] == "x") and
                        not (phonemes_string[-1] == "i" and word[-1] == "y") and
                        not (word[0] == "k" and phonemes_string[1] not in "eéèẹẽẻêếềệễểiíìịĩỉyýỳỵỹỷ") and
                        not (word[0] == 'g' and word[1] == 'i')):  # and len(word) < len(vietdict[phonemes_string]):
                    vietdict[phonemes_string] = word

    return vietdict


def read_vn_dict(dictpath):
    """
        this dict is the common dict: dict[word] = phoneme, like dict['địt'] = '5 d i t'
    :param dictpath:
    :return:
    """
    vietdict = {item: f"{tone_for_unvoiced_sound} {item}" for item in unvoiced_dict}

    with open(dictpath, "r") as f:
        for line in f.readlines():
            if line and line.strip() and line[0] != "#" and "xx" not in line and "pau\n" not in line:
                if "completelynotavailable" not in line or ("completelynotavailable" in line and len(line.strip().replace(" completelynotavailable", "").strip().split(" ")) > 1):
                    line = line.strip().replace(" notavailable", "").replace(" completelynotavailable", "")
                    # temp = line.strip().split(" ")
                    # word, phonemes = temp[0], temp[1:]
                    word, phonemes = line.split(" ", maxsplit=1)

                    vietdict[word] = phonemes

    for key, value in map_phone_to_unvoice.items():
        vietdict[f"({value})"] = key

    # Big note: This is for future mid (and some south region) dialects.
    # Since north dialect typically pronoun "sờ" as "xờ", this have absolutely no effect
    vietdict["(sờ)"] = "6 s"
    vietdict[" "] = "space"
    # endnote kinhph

    return vietdict


def get_phoneme_transcription(word_and_transcription, vietdict):
    """
        Convert (oov) word with syllable transcription to phonemes
        For example: chelsea chen xi -> 0 ch e nz 0 s i
    :param word_and_transcription:
    :param vietdict:
    :return:
    """
    # temp = word_and_transcription.strip().split(" ")
    temp = word_and_transcription.strip().replace(" ", ", ,").replace(", ,", ",", 1).split(",")  # By replacing space with ", ," I introduced the " " (space) syllable into the problem. This will facilitate the process of converting sequence of phoneme to syllable later
    word, syllables = temp[0], temp[1:]

    result = []
    nb_phoneme = 0

    for syllable in syllables:
        try:
            result.append(vietdict[syllable])
            nb_phoneme += len(vietdict[syllable].split(" ")[1:])
        except Exception as e:
            print(word_and_transcription, e)

    return word + " " + " ".join(result), nb_phoneme
