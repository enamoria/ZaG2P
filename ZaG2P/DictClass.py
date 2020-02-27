# -*- coding: utf-8 -*-

""" Created on 3:39 PM, 7/23/19
    @author: ngunhuconchocon
    @brief: Пролетарии всех стран, соединяйтесь!
"""

from __future__ import print_function, division, absolute_import

import torchtext.data as data
import random
import pdb
from . import constant
from .utils import read_dict, read_vn_dict, get_phoneme_transcription, nucleus


# Demo on Cargie Merlon dataset
class CMUDict(data.Dataset):
    def __init__(self, data_lines, g_field, p_field):
        fields = [('grapheme', g_field), ('phoneme', p_field)]
        examples = []  # maybe ignore '...-1' grapheme
        for line in data_lines:
            grapheme, phoneme = line.split(maxsplit=1)
            examples.append(data.Example.fromlist([grapheme, phoneme],
                                                  fields))
        self.sort_key = lambda x: len(x.grapheme)
        super(CMUDict, self).__init__(examples, fields)

    @classmethod
    def splits(cls, path, g_field, p_field, seed=None):
        if seed is not None:
            random.seed(seed)
        with open(path) as f:
            lines = f.readlines()
        random.shuffle(lines)
        train_lines, val_lines, test_lines = [], [], []
        for i, line in enumerate(lines):
            if i % 20 == 0:
                val_lines.append(line)
            elif i % 20 < 3:
                test_lines.append(line)
            else:
                train_lines.append(line)
        train_data = cls(train_lines, g_field, p_field)
        val_data = cls(val_lines, g_field, p_field)
        test_data = cls(test_lines, g_field, p_field)
        return train_data, val_data, test_data


# Vietnamese Dictionary
class VNDict(data.Dataset):
    def __init__(self, data_lines, g_field, p_field, test=False):
        fields = [('grapheme', g_field), ('phoneme', p_field)]
        examples = []

        for line in data_lines:
            if line and line.strip() and line[0] != "#":
                if test:
                    grapheme, phoneme = line.strip().strip(","), "xxxxx"
                else:
                    grapheme, phoneme = line.strip().split(" ", 1)
                examples.append(data.Example.fromlist([grapheme, phoneme], fields))

        self.sort_key = lambda x: len(x.grapheme)
        super(VNDict, self).__init__(examples, fields)

    @classmethod
    def splits(cls, path, g_field, p_field, seed=None, drop_headline=18, datatype="phoneme"):
        if datatype != "phoneme" and datatype != "syllable":
            print("unsupport datatype. Must be either phoneme or syllable. Use phoneme by default")
            datatype = "phoneme"

        if seed:
            random.seed(seed)
        else:
            random.seed(1)

        # with open(path) as f:
        #     lines = f.readlines()
        lines, all_data, total_phoneme = cls.get_dict(path, drop_headline, datatype)

        random.shuffle(lines)
        train_lines, val_lines, test_lines = [], [], []
        for i, line in enumerate(lines):
            if i % 20 == 0:
                val_lines.append(line)
            elif i % 20 < 2:
                test_lines.append(line)
            else:
                train_lines.append(line)
        # for line in lines[-100:]:
        #     test_lines.append(line)

        train_data = cls(train_lines, g_field, p_field)
        val_data = cls(val_lines, g_field, p_field)
        test_data = cls(test_lines, g_field, p_field)
        combined_data = cls(lines, g_field, p_field)

        return train_data, val_data, test_data, combined_data, all_data, total_phoneme

    @staticmethod
    def get_dict(dictpath, drop_headline, datatype):
        """
            This method combine a dictionary with format <syllable> [<tone> [<phoneme> ...] ] to <syllable> [ [<phoneme_tone> ...] ..]
            For example: chelsea 0 ch e nz 0 s i -> chelsea ch e0 nz0 s i0
        :param datatype:
        :param drop_headline: Number of head lines of the dict that will be dropped.  since vn.dict start with some punctuation and we don't need it, we will drop it.
        :param dictionary: vn.dict format :D
        :return: combined dictioanry: vn.dict format
        """
        vn_dict = []
        vn_dict_original = {}  # This dict is the dict we usually use: tone and phoneme are separated
        vietdict = read_vn_dict("ZaG2P/tts_dict_prepare/vn.dict")  # (single) syllable -> phonemes

        total_phoneme = 0

        with open(dictpath, "r") as f:
            for line in f.readlines()[drop_headline:]:
                if line and line.strip() and line[0] != "#" and "xxxxx" not in line:
                    # kinhph update: input is always syllable. datatype specify whether to convert it to phoneme or not
                    if datatype == "phoneme":
                        line, nb_phoneme = get_phoneme_transcription(line, vietdict)
                        total_phoneme += nb_phoneme

                    # end of editing kinhph

                    items = line.strip().split(" ")

                    if len(items) > 1:  # this word had been transcribed into phonemes
                        syllable = items[0]

                        # Add tone to phonemes
                        if items[1].isdigit():
                            vn_dict_original[syllable] = items[1:]

                            curr_tone = items[1]
                            phonemes = items[2:]
                            result = []

                            # total = 0
                            # total_tone = 1
                            for phoneme in phonemes:
                                # if phoneme in constant.coda_nucleus_and_semivowel:  # both nucleus and coda got tone info
                                if phoneme != "space":
                                    if phoneme in constant.nucleus:  # only nucleus got tone info
                                        result.append(phoneme + curr_tone)
                                        # total += 1
                                    elif phoneme.isdigit():
                                        curr_tone = phoneme
                                        # total_tone += 1
                                    elif curr_tone == constant.tone_for_unvoiced_sound:
                                        result.append(phoneme + curr_tone)
                                    else:
                                        result.append(phoneme)
                                else:
                                    result.append(phoneme)

                            # if total >= 2 and total > total_tone:
                            #     print(f"{line}\t{total_tone}\t{total}")

                            str_result = " ".join(result)
                            # vn_dict.append(f"{syllable} {str_result}")
                            vn_dict.append(f"{syllable} {str_result}")
                        else:  # OOV transcribe from doccano syllables
                            str_result = " ".join(items[1:])
                            # vn_dict.append(f"{syllable} {str_result}")
                            vn_dict.append(f"{syllable} {str_result}")

        return vn_dict, vn_dict_original, total_phoneme
