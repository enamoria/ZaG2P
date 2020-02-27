# -*- coding: utf-8 -*-

""" Created on 9:43 AM, 9/4/19
    @author: ngunhuconchocon
    @brief: Пролетарии всех стран, соединяйтесь! да здравствует наша советская родина
"""

from setuptools import setup

setup(
    name="ZaG2P",
    version="1.0.0",
    author="ngunhuconchocon",
    author_email="phanhuykinh@gmail.com",
    description="Convert non-Vietnamese word to Vietnamese phonemes/syllables",
    # license="BSD",
    url="https://github.com/enamoria/ZaG2P",
    install_requires=[
        # 'python-Levenshtein @ git+https://github.com/ztane/python-Levenshtein',
        'torch==1.1.0', 'torchtext==0.3.1', 'dill', 'visdom', 'textdistance'
    ],
    packages=['ZaG2P'],
    include_package_data=True
)
