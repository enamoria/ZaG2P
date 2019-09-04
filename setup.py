# -*- coding: utf-8 -*-

""" Created on 9:43 AM, 9/4/19
    @author: ngunhuconchocon
    @brief: Пролетарии всех стран, соединяйтесь! да здравствует наша советская родина
"""

from setuptools import setup

setup(
    name="pyEval",
    version="0.0.1",
    author="ngunhuconchocon",
    description="Convert non-Vietnamese word to Vietnamese phonemes/syllables",
    # license="BSD",
    # url="https://github.com/enamoria/ZaG2P",
    install_requires=[
          'Levenshtein', 'torch>=1.1.0', 'torchtext>=0.3.1'
      ],
    dependency_links=[
        "git+https://github.com/ztane/python-Levenshtein#egg=Levenshtein"
    ],
    packages=['ZaG2P'],
    include_package_data=True
)