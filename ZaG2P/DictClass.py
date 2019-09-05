# -*- coding: utf-8 -*-

""" Created on 3:39 PM, 7/23/19
    @author: ngunhuconchocon
    @brief: Пролетарии всех стран, соединяйтесь!
"""

from __future__ import print_function, division, absolute_import

import torchtext.data as data
import random
from . import constant


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
    def __init__(self, data_lines, g_field, p_field):
        fields = [('grapheme', g_field), ('phoneme', p_field)]
        examples = []

        for line in data_lines:
            if line and line.strip() and line[0] != "#":
                grapheme, phoneme = line.split(" ", 1)
                examples.append(data.Example.fromlist([grapheme, phoneme], fields))

        self.sort_key = lambda x: len(x.grapheme)
        super(VNDict, self).__init__(examples, fields)

    @classmethod
    def splits(cls, path, g_field, p_field, seed=None, drop_headline=18):
        random.seed(seed)

        # with open(path) as f:
        #     lines = f.readlines()
        lines, all_data = cls.get_dict(path, drop_headline)

        # random.shuffle(lines)
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

        return train_data, val_data, test_data, all_data

    @staticmethod
    def get_dict(dictpath, drop_headline):
        """
            This method combine a dictionary with format <syllable> [<tone> [<phoneme> ...] ] to <syllable> [ [<phoneme_tone> ...] ..]
            For example: chelsea 0 ch e nz 0 s i -> chelsea ch e0 nz0 s i0
        :param drop_headline: Number of head lines of the dict that will be dropped.  since vn.dict start with some punctuation and we don't need it, we will drop it.
        :param dictionary: vn.dict format :D
        :return: combined dictioanry: vn.dict format
        """
        vn_dict = []
        vn_dict_original = {}  # This dict is the dict we usually use: tone and phoneme are separated

        with open(dictpath, "r") as f:
            for line in f.readlines()[drop_headline:]:
                if line and line.strip() and line[0] != "#":
                    # if "villa" in line:
                    #     print(1)
                    items = line.strip().split(" ")

                    if len(items) > 1:  # this word had been transcribed into phonemes
                        syllable = items[0]

                        # Add tone to phonemes
                        if items[1].isdigit():
                            vn_dict_original[syllable] = items[1:]

                            curr_tone = items[1]
                            phonemes = items[2:]
                            result = []
                            for phoneme in phonemes:
                                if phoneme in constant.coda_nucleus_and_semivowel:
                                    result.append(phoneme + curr_tone)
                                elif phoneme.isdigit():
                                    curr_tone = phoneme
                                    # result.append(" ")
                                elif curr_tone == constant.tone_for_unvoiced_sound:
                                    result.append(phoneme + curr_tone)
                                else:
                                    result.append(phoneme)

                            str_result = " ".join(result)
                            # vn_dict.append(f"{syllable} {str_result}")
                            vn_dict.append("{syllable} {str_result}".format(syllable=syllable, str_result=str_result))
                        else:  # OOV transcribe from doccano syllables
                            str_result = " ".join(items[1:])
                            # vn_dict.append(f"{syllable} {str_result}")
                            vn_dict.append("{syllable} {str_result}".format(syllable=syllable, str_result=str_result))

        return vn_dict, vn_dict_original
