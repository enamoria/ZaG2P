# -*- coding: utf-8 -*-

""" Created on 11:12 AM, 7/24/19
    @author: ngunhuconchocon
    @brief: Пролетарии всех стран, соединяйтесь!
"""
import yaml
import os

project_root = os.path.abspath(os.path.dirname(__file__))
coda_nucleus_and_semivowel = ['iz', 'pc', 'nz', 'tc', 'ngz', 'kc', 'uz', 'mz', 'aa', 'ee', 'ea', 'oa', 'aw', 'ie', 'uo', 'a', 'wa', 'oo', 'e', 'i',
                              'o', 'u', 'ow', 'uw', 'w']

nucleuses = ['o', 'oa', 'oo', 'u', 'a', 'aa', 'aw', 'ow', 'uw', 'ie', 'uo', 'wa', 'e', 'ea', 'ee', 'i']

tone_for_unvoiced_sound = "6"
# This is for 8tone dictionary
# coda_nucleus_and_semivowel = ['ngmz', 'ie', 'uo', 'nhz', 'tz', 'pau', 'pz', 'nz', 'aX', 'WX', 'ngz', 'E', 'jz', 'OX', 'O', 'EX', 'W', 'wz', 'kcz', 'a', 'e', 'i', 'o', 'MW', 'u', 'w', 'kz', 'z', 'kpz', 'mz']

# Parser for some hyperparameter in training G2P
with open(os.path.join(project_root, "config.yml"), "r") as f:
    parser = yaml.load(f)
