# -*- coding: utf-8 -*-

""" Created on 10:16 AM, 8/15/19
    @author: ngunhuconchocon
    @brief: Пролетарии всех стран, соединяйтесь! да здравствует наша советская родина
"""

from __future__ import print_function, division, absolute_import

import pdb
import dill as pickle
# import cloudpickle as pickle
import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchtext.data as data

# Based on https://github.com/SeanNaren/deepspeech.pytorch/blob/master/decoder.py.
import Levenshtein  # https://github.com/ztane/python-Levenshtein/

from .DictClass import VNDict
from .models import G2P
from .constant import parser, project_root

from . import utils
# import utils
import logging
from datetime import datetime

logging.basicConfig(level=logging.DEBUG)

parser['intermediate_path'] = 'intermediate/g2p_vi/'  # path to save models
parser['beam_size'] = 10  # size of beam for beam-search
parser['d_embed'] = 350  # embedding dimension
parser['d_hidden'] = 350  # hidden dimension
parser['epochs'] = 10
parser['max_len'] = 10  # max length of grapheme/phoneme sequences
parser['lr'] = 0.01
parser['lr_min'] = 1e-5

args = argparse.Namespace(**parser)
args.cuda = args.cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


def phoneme_error_rate(p_seq1, p_seq2, batch):
    _g_field = batch.dataset.fields['grapheme']
    _p_field = batch.dataset.fields['phoneme']

    p_vocab = set(p_seq1 + p_seq2)
    p2c = dict(zip(p_vocab, range(len(p_vocab))))
    c_seq1 = [chr(p2c[p]) for p in p_seq1]
    c_seq2 = [chr(p2c[p]) for p in p_seq2]

    # print(' '.join([p_field.vocab.itos[p] for p in p_seq1]),
    #       ' '.join([p_field.vocab.itos[p] for p in p_seq2]))

    # return Levenshtein.distance(''.join(c_seq1), ''.join(c_seq2)) / len(c_seq2)
    try:
        return Levenshtein.distance(' '.join([p_field.vocab.itos[p] for p in p_seq1]),
                                    ' '.join([p_field.vocab.itos[p] for p in p_seq2])) / float(len(p_seq2))
    except Exception as e:
        return 0.99
        pdb.set_trace()


def adjust_learning_rate(optimizer, lr_decay):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_decay


def train(config, train_iter, model, criterion, optimizer, epoch, test_iter=None):
    global iteration, n_total, train_loss, n_bad_loss
    global init, best_val_loss, stop

    print("=> EPOCH {}".format(epoch))

    train_iter.init_epoch()
    for batch in train_iter:
        iteration += 1
        model.train()

        output, _, __ = model(batch.grapheme, batch.phoneme[:-1].detach())
        target = batch.phoneme[1:]
        loss = criterion(output.view(output.size(0) * output.size(1), -1),
                         target.view(target.size(0) * target.size(1)))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), config.clip, 'inf')
        optimizer.step()

        n_total += batch.batch_size
        train_loss += loss.data * batch.batch_size

        if iteration % config.log_every == 0:
            train_loss /= n_total
            val_loss = validate(val_iter, model, criterion)
            print("   % Time: {:5.0f} | Iteration: {:5} | Batch: {:4}/{}"
                  " | Train loss: {:.4f} | Val loss: {:.4f}"
                  .format(time.time() - init, iteration, train_iter.iterations,
                          len(train_iter), train_loss, val_loss))

            plotter.plot('loss', 'train', 'Class Loss', epoch, float(train_loss))
            plotter.plot('loss', 'val', 'Class Loss', epoch, float(val_loss))

            # test for val_loss improvement
            n_total = train_loss = 0
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                n_bad_loss = 0
                torch.save(model.state_dict(), config.best_model)
            else:
                n_bad_loss += 1

            if n_bad_loss == config.n_bad_loss:
                best_val_loss = val_loss
                n_bad_loss = 0
                adjust_learning_rate(optimizer, config.lr_decay)
                new_lr = optimizer.param_groups[0]['lr']
                print("=> Adjust learning rate to: {}".format(new_lr))
                if new_lr < config.lr_min:
                    stop = True
                    break

    if epoch % 3 == 0 and epoch > 5 and test_iter:
        test(test_iter, model, criterion)


def validate(val_iter, model, criterion):
    model.eval()
    val_loss = 0
    val_iter.init_epoch()
    for batch in val_iter:
        output, _, __ = model(batch.grapheme, batch.phoneme[:-1])
        target = batch.phoneme[1:]
        loss = criterion(output.squeeze(1), target.squeeze(1))
        # val_loss += loss.data[0] * batch.batch_size
        val_loss += loss.data * batch.batch_size
    return val_loss / len(val_iter.dataset)


def test(test_iter, model, criterion):
    model.eval()
    test_iter.init_epoch()
    test_per = test_wer = 0
    for batch in test_iter:
        # print(batch.grapheme)
        # pdb.set_trace()
        output = model(batch.grapheme).data.tolist()
        target = batch.phoneme[1:].squeeze(1).data.tolist()
        # calculate per, wer here
        per = phoneme_error_rate(output, target, batch)
        wer = int(output != target)
        test_per += per  # batch_size = 1
        test_wer += wer

    test_per = test_per / len(test_iter.dataset) * 100
    test_wer = test_wer / len(test_iter.dataset) * 100
    print("Phoneme error rate (PER): {:.2f}\nWord error rate (WER): {:.2f}"
          .format(test_per, test_wer))


def show(batch, model, f=None):
    assert batch.batch_size == 1
    g_field = batch.dataset.fields['grapheme']
    p_field = batch.dataset.fields['phoneme']
    prediction = model(batch.grapheme).data.tolist()[:-1]
    grapheme = batch.grapheme.squeeze(1).data.tolist()[1:][::-1]
    phoneme = batch.phoneme.squeeze(1).data.tolist()[1:-1]

    ground_truth_grapheme = ''.join([g_field.vocab.itos[g] for g in grapheme])
    ground_truth_phoneme = ' '.join([p_field.vocab.itos[p] for p in phoneme])
    prediction = ' '.join([p_field.vocab.itos[p] for p in prediction])

    if ground_truth_phoneme != prediction:
        if f:
            f.write(u"> {}\n= {}\n< {}\n\n".format(ground_truth_grapheme, ground_truth_phoneme, prediction))

        return 0
    return 1  # if correct
    # else:
    #     print(f"{ground_truth_grapheme}, {ground_truth_phoneme}, {prediction}")
    # f.write(u"> {}\n= {}\n< {}\n\n".format(
    #     ''.join([g_field.vocab.itos[g] for g in grapheme]),
    #     ' '.join([p_field.vocab.itos[p] for p in phoneme]),
    #     ' '.join([p_field.vocab.itos[p] for p in prediction])).encode("utf8"))


if __name__ == "__main__":
    global plotter
    plotter = utils.VisdomLinePlotter(env_name='ZaG2P vietnamese phonemes')

    g_field = data.Field(init_token='<s>', tokenize=(lambda x: list(x.split()[0])[::-1]))
    p_field = data.Field(init_token='<os>', eos_token='</os>',
                         tokenize=(lambda x: x.split()))

    # g_field = data.Field(tokenize=(lambda x: list(x.split()[0])[::-1]))
    # p_field = data.Field(tokenize=(lambda x: x.split()))

    # filepath = os.path.join(args.data_path, 'vn.dict')
    # filepath = os.path.join(args.vi_data_path, 'oov.vn.dict_g2p_1_4')
    # filepath = os.path.join(project_root, os.path.join(args.vi_data_path, 'oov.vn.dict'))
    filepath = os.path.join(project_root, os.path.join(args.vi_data_path, 'oov_syllable_new_type_1'))
    train_data, val_data, test_data, combined_data, all_data = VNDict.splits(filepath, g_field, p_field, args.seed)

    g_field.build_vocab(train_data, val_data, test_data)
    p_field.build_vocab(train_data, val_data, test_data)

    # device = None if args.cuda else -1  # None is current gpu
    device = "cuda" if args.cuda else -1
    train_iter = data.BucketIterator(train_data, batch_size=args.batch_size,
                                     repeat=False, device=device)
    val_iter = data.Iterator(val_data, batch_size=1,
                             train=False, sort=False, device=device)
    test_iter = data.Iterator(test_data, batch_size=1,
                              train=False, shuffle=True, device=device)
    # test_iter = data.Iterator(["tivi"], batch_size=1,
    #                           train=False, shuffle=True, device=device)

    config = args
    config.g_size = len(g_field.vocab)
    config.p_size = len(p_field.vocab)

    print(config.g_size, config.p_size)
    config.best_model = os.path.join(project_root, os.path.join(config.intermediate_path, "best_model_adagrad_attn.pth"))

    if not os.path.isdir(config.intermediate_path):
        os.system("mkdir -p {intermediate_path}".format(intermediate_path=config.intermediate_path))

    with open(os.path.join(project_root, os.path.join(args.intermediate_path, "gp_fields.pkl")), "wb") as f:
        pickle.dump({
            "g_field": g_field,
            "p_field": p_field,
            "config": config,
            "g2p_training_dict": all_data
        }, f, protocol=2)

    model = G2P(config)
    criterion = nn.NLLLoss()
    if config.cuda:
        model.cuda()
        criterion.cuda()
    optimizer = optim.Adagrad(model.parameters(), lr=config.lr)  # use Adagrad

    # test(test_iter, model, criterion)
    # test_iter.init_epoch()

    if 1 == 1:  # change to True to train
        iteration = n_total = train_loss = n_bad_loss = 0
        stop = False
        best_val_loss = 10
        init = time.time()
        for epoch in range(1, config.epochs + 1):
            if epoch == 15:
                # pdb.set_trace()
                print(1)
            train(config, train_iter, model, criterion, optimizer, epoch, test_iter)
            if stop:
                break

    model.load_state_dict(torch.load(config.best_model))
    test(test_iter, model, criterion)
    test_iter.init_epoch()

    test_on_train_iter = data.Iterator(combined_data, batch_size=1,
                                       train=False, shuffle=True, device=device)

    count_correct = 0

    filepath = os.path.join(project_root, f"logs/output_{parser['d_embed']}_{parser['d_hidden']}_epoch{parser['epochs']}_lrdecay{parser['lr_decay']}")
    os.system(f"mkdir -p " + os.path.join(project_root + "logs"))

    with open(filepath, "wb") as f_out:
        with open(os.path.join(project_root, "test_train_G2P.csv"), "w") as ff:
            for i, batch in enumerate(test_on_train_iter):
                count_correct += show(batch, model, ff)

        print(f"{count_correct}/{len(test_on_train_iter)}")
        f_out.write(f"{count_correct}/{len(test_on_train_iter)}\n".encode("utf8"))

        count_correct = 0
        with open(os.path.join(project_root, "test_test_G2P.csv"), "w") as ff:
            for i, batch in enumerate(test_iter):
                count_correct += show(batch, model, ff)
            print(f"{count_correct}/{len(test_iter)}")
        f_out.write(f"{count_correct}/{len(test_iter)}".encode("utf8"))

    logging.info("Done at {}".format(datetime.now().strftime("%Y-%m-%d %H:%M")))

# Wayne Rooney và Cristiano Ronaldo đều là những tên tuổi lớvín đã từng thi đấu cho Manchester United. Hiện tại, Ronaldo đang thi đấu tại Series A,
# Rooney đến Mỹ thi đấu tại giải bóng đá nhà nghề MLS. Trong khi cựu số 7 vẫn giữ được sự sắc sảo trên hàng công Real Madrid và Juventus và liên tiếp
# lên đỉnh Châu Âu, Rooney thi đấu mờ nhạt hơn và không giành được danh hiệu nào đáng kể ngoài UEFA Europa League, giải đấu mà anh thường xuyên ngồi ghế dự bị
